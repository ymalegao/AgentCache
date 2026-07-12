#!/usr/bin/env python3
"""
Build deterministic code-eval tasks for data/python_code_eval.jsonl, with
VERIFIED expected_stdout.

Each task ships a reference solution. Before writing a task, this script runs the
reference TWICE in a subprocess and requires byte-identical stdout, then stores
that real stdout as the golden `expected_stdout`. Nothing is hand-guessed, so the
goldens can never drift from what correct code actually prints.

Re-runnable / append-only:
  - The next code_NNN id is computed from the existing file (no manual indexing).
  - Tasks are deduped by their exact `user` prompt, so re-running only appends
    tasks not already in the file. Add new tasks to TASKS and run again.

To add tasks: append (slug, user_prompt, must_include_substr, reference_solution)
tuples to the TASKS list below. Keep solutions stdlib-only and deterministic
(no randomness / time / network / set-ordering-dependent output).

Usage:
    # verify new tasks without writing
    python agentcache_compression/generate_code_eval_tasks.py --dry-run

    # verify and append new tasks to the eval file
    python agentcache_compression/generate_code_eval_tasks.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_DEFAULT_DATA = _EXP / "data" / "python_code_eval.jsonl"

_ID_RE = re.compile(r"code_(\d+)")


# ---------------------------------------------------------------------------
# Tasks.  Tuple: (slug, user_prompt, must_include_substr, reference_solution)
#   - slug                : short label, only for console output (not stored)
#   - user_prompt         : the task text shown to the model
#   - must_include_substr : substring required in the model's code (checks gate)
#   - reference_solution  : a correct stdlib-only program; its stdout becomes the
#                           golden expected_stdout
# ---------------------------------------------------------------------------

TASKS: list[tuple[str, str, str, str]] = [
    ("sum_digits",
     "Write a function sum_digits(n) that returns the sum of the digits of a non-negative integer. Print sum_digits(12345).",
     "def sum_digits",
     "def sum_digits(n):\n    return sum(int(d) for d in str(n))\nprint(sum_digits(12345))\n"),

    ("int_to_roman",
     "Write a function int_to_roman(n) that converts an integer (1-3999) to a Roman numeral string. Print int_to_roman(2024).",
     "def int_to_roman",
     ("def int_to_roman(n):\n"
      "    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),\n"
      "            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
      "    out = []\n"
      "    for v, sym in vals:\n"
      "        while n >= v:\n"
      "            out.append(sym); n -= v\n"
      "    return ''.join(out)\n"
      "print(int_to_roman(2024))\n")),

    ("fizzbuzz",
     "Write code that prints the numbers 1 to 15, one per line, but prints 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, and 'FizzBuzz' for multiples of both.",
     "FizzBuzz",
     ("for i in range(1, 16):\n"
      "    if i % 15 == 0:\n"
      "        print('FizzBuzz')\n"
      "    elif i % 3 == 0:\n"
      "        print('Fizz')\n"
      "    elif i % 5 == 0:\n"
      "        print('Buzz')\n"
      "    else:\n"
      "        print(i)\n")),

    ("collatz_steps",
     "Write a function collatz_steps(n) that returns the number of steps to reach 1 in the Collatz sequence (n->n/2 if even, n->3n+1 if odd). Print collatz_steps(27).",
     "def collatz_steps",
     ("def collatz_steps(n):\n"
      "    steps = 0\n"
      "    while n != 1:\n"
      "        n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
      "        steps += 1\n"
      "    return steps\n"
      "print(collatz_steps(27))\n")),

    ("mod_pow",
     "Write a function mod_pow(base, exp, mod) that computes (base**exp) % mod efficiently. Print mod_pow(2, 10, 1000).",
     "def mod_pow",
     "def mod_pow(base, exp, mod):\n    return pow(base, exp, mod)\nprint(mod_pow(2, 10, 1000))\n"),

    ("nth_prime",
     "Write a function nth_prime(n) that returns the nth prime number (1-indexed, nth_prime(1)=2). Print nth_prime(10).",
     "def nth_prime",
     ("def nth_prime(n):\n"
      "    primes = []\n"
      "    cand = 2\n"
      "    while len(primes) < n:\n"
      "        if all(cand % p for p in primes if p * p <= cand):\n"
      "            primes.append(cand)\n"
      "        cand += 1\n"
      "    return primes[-1]\n"
      "print(nth_prime(10))\n")),

    ("sum_primes_below",
     "Write a function sum_primes_below(n) that returns the sum of all primes strictly less than n. Print sum_primes_below(20).",
     "def sum_primes_below",
     ("def sum_primes_below(n):\n"
      "    def is_prime(x):\n"
      "        if x < 2:\n"
      "            return False\n"
      "        i = 2\n"
      "        while i * i <= x:\n"
      "            if x % i == 0:\n"
      "                return False\n"
      "            i += 1\n"
      "        return True\n"
      "    return sum(x for x in range(n) if is_prime(x))\n"
      "print(sum_primes_below(20))\n")),

    ("trailing_zeros",
     "Write a function trailing_zeros(n) that returns the number of trailing zeros in n! (n factorial). Print trailing_zeros(100).",
     "def trailing_zeros",
     ("def trailing_zeros(n):\n"
      "    count = 0\n"
      "    p = 5\n"
      "    while p <= n:\n"
      "        count += n // p\n"
      "        p *= 5\n"
      "    return count\n"
      "print(trailing_zeros(100))\n")),

    ("is_perfect",
     "Write a function is_perfect(n) that returns True if n equals the sum of its proper divisors. Print is_perfect(28).",
     "def is_perfect",
     ("def is_perfect(n):\n"
      "    return n == sum(d for d in range(1, n) if n % d == 0)\n"
      "print(is_perfect(28))\n")),

    ("lcm",
     "Write a function lcm(a, b) that returns the least common multiple of two positive integers. Print lcm(4, 6).",
     "def lcm",
     ("from math import gcd\n"
      "def lcm(a, b):\n"
      "    return a * b // gcd(a, b)\n"
      "print(lcm(4, 6))\n")),

    ("digital_root",
     "Write a function digital_root(n) that repeatedly sums the digits of n until a single digit remains, and returns it. Print digital_root(9875).",
     "def digital_root",
     ("def digital_root(n):\n"
      "    while n >= 10:\n"
      "        n = sum(int(d) for d in str(n))\n"
      "    return n\n"
      "print(digital_root(9875))\n")),

    ("hamming_distance",
     "Write a function hamming_distance(a, b) that returns the number of positions at which two equal-length strings differ. Print hamming_distance('karolin', 'kathrin').",
     "def hamming_distance",
     ("def hamming_distance(a, b):\n"
      "    return sum(1 for x, y in zip(a, b) if x != y)\n"
      "print(hamming_distance('karolin', 'kathrin'))\n")),

    ("run_length_encode",
     "Write a function run_length_encode(s) that encodes a string as character followed by its run count, e.g. 'aaabbbcccd' -> 'a3b3c3d1'. Print run_length_encode('aaabbbcccd').",
     "def run_length_encode",
     ("def run_length_encode(s):\n"
      "    if not s:\n"
      "        return ''\n"
      "    out = []\n"
      "    prev = s[0]\n"
      "    count = 1\n"
      "    for ch in s[1:]:\n"
      "        if ch == prev:\n"
      "            count += 1\n"
      "        else:\n"
      "            out.append(f'{prev}{count}')\n"
      "            prev = ch\n"
      "            count = 1\n"
      "    out.append(f'{prev}{count}')\n"
      "    return ''.join(out)\n"
      "print(run_length_encode('aaabbbcccd'))\n")),

    ("matrix_multiply",
     "Write a function matrix_multiply(a, b) that multiplies two 2D matrices (lists of lists). Print matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]).",
     "def matrix_multiply",
     ("def matrix_multiply(a, b):\n"
      "    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))\n"
      "             for j in range(len(b[0]))] for i in range(len(a))]\n"
      "print(matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]))\n")),

    ("rotate_right",
     "Write a function rotate_right(lst, k) that rotates a list to the right by k positions. Print rotate_right([1, 2, 3, 4, 5], 2).",
     "def rotate_right",
     ("def rotate_right(lst, k):\n"
      "    if not lst:\n"
      "        return lst\n"
      "    k %= len(lst)\n"
      "    return lst[-k:] + lst[:-k] if k else lst[:]\n"
      "print(rotate_right([1, 2, 3, 4, 5], 2))\n")),

    ("chunk",
     "Write a function chunk(lst, n) that splits a list into consecutive chunks of size n (the last chunk may be smaller). Print chunk([1, 2, 3, 4, 5, 6, 7], 3).",
     "def chunk",
     ("def chunk(lst, n):\n"
      "    return [lst[i:i + n] for i in range(0, len(lst), n)]\n"
      "print(chunk([1, 2, 3, 4, 5, 6, 7], 3))\n")),

    ("most_common",
     "Write a function most_common(lst) that returns the most frequently occurring element. Print most_common([1, 2, 2, 3, 3, 3]).",
     "def most_common",
     ("from collections import Counter\n"
      "def most_common(lst):\n"
      "    return Counter(lst).most_common(1)[0][0]\n"
      "print(most_common([1, 2, 2, 3, 3, 3]))\n")),

    ("second_largest",
     "Write a function second_largest(lst) that returns the second largest distinct value in a list. Print second_largest([3, 1, 4, 1, 5, 9, 2, 6]).",
     "def second_largest",
     ("def second_largest(lst):\n"
      "    return sorted(set(lst))[-2]\n"
      "print(second_largest([3, 1, 4, 1, 5, 9, 2, 6]))\n")),

    ("is_balanced",
     "Write a function is_balanced(s) that returns True if all brackets (), [], {} in a string are correctly balanced and nested. Print is_balanced('(a[b]{c})') and is_balanced('(]').",
     "def is_balanced",
     ("def is_balanced(s):\n"
      "    pairs = {')': '(', ']': '[', '}': '{'}\n"
      "    stack = []\n"
      "    for ch in s:\n"
      "        if ch in '([{':\n"
      "            stack.append(ch)\n"
      "        elif ch in pairs:\n"
      "            if not stack or stack.pop() != pairs[ch]:\n"
      "                return False\n"
      "    return not stack\n"
      "print(is_balanced('(a[b]{c})'))\n"
      "print(is_balanced('(]'))\n")),

    ("count_bits",
     "Write a function count_bits(n) that returns the number of 1 bits in the binary representation of a non-negative integer. Print count_bits(255).",
     "def count_bits",
     "def count_bits(n):\n    return bin(n).count('1')\nprint(count_bits(255))\n"),

    ("to_binary",
     "Write a function to_binary(n) that returns the binary representation of a non-negative integer as a string without the '0b' prefix. Print to_binary(42).",
     "def to_binary",
     "def to_binary(n):\n    return bin(n)[2:]\nprint(to_binary(42))\n"),

    ("from_binary",
     "Write a function from_binary(s) that converts a binary string to its integer value. Print from_binary('101010').",
     "def from_binary",
     "def from_binary(s):\n    return int(s, 2)\nprint(from_binary('101010'))\n"),

    ("pascal_row",
     "Write a function pascal_row(n) that returns the nth row (0-indexed) of Pascal's triangle as a list. Print pascal_row(4).",
     "def pascal_row",
     ("def pascal_row(n):\n"
      "    row = [1]\n"
      "    for k in range(1, n + 1):\n"
      "        row.append(row[-1] * (n - k + 1) // k)\n"
      "    return row\n"
      "print(pascal_row(4))\n")),

    ("is_armstrong",
     "Write a function is_armstrong(n) that returns True if n equals the sum of its own digits each raised to the power of the number of digits. Print is_armstrong(153).",
     "def is_armstrong",
     ("def is_armstrong(n):\n"
      "    digits = str(n)\n"
      "    p = len(digits)\n"
      "    return n == sum(int(d) ** p for d in digits)\n"
      "print(is_armstrong(153))\n")),

    ("reverse_words",
     "Write a function reverse_words(s) that reverses the order of words in a string (single spaces). Print reverse_words('the sky is blue').",
     "def reverse_words",
     ("def reverse_words(s):\n"
      "    return ' '.join(s.split()[::-1])\n"
      "print(reverse_words('the sky is blue'))\n")),

    ("count_unique_chars",
     "Write a function count_unique_chars(s) that returns the number of distinct characters in a string. Print count_unique_chars('abracadabra').",
     "def count_unique_chars",
     ("def count_unique_chars(s):\n"
      "    return len(set(s))\n"
      "print(count_unique_chars('abracadabra'))\n")),

    ("find_missing",
     "Write a function find_missing(nums) that, given a list containing n distinct numbers from 0 to n, returns the one missing number. Print find_missing([3, 0, 1]).",
     "def find_missing",
     ("def find_missing(nums):\n"
      "    n = len(nums)\n"
      "    return n * (n + 1) // 2 - sum(nums)\n"
      "print(find_missing([3, 0, 1]))\n")),

    ("fib_list",
     "Write a function fib_list(n) that returns a list of the first n Fibonacci numbers (starting 0, 1). Print fib_list(8).",
     "def fib_list",
     ("def fib_list(n):\n"
      "    out = []\n"
      "    a, b = 0, 1\n"
      "    for _ in range(n):\n"
      "        out.append(a)\n"
      "        a, b = b, a + b\n"
      "    return out\n"
      "print(fib_list(8))\n")),

    ("prime_factors",
     "Write a function prime_factors(n) that returns the list of prime factors of n in ascending order (with multiplicity). Print prime_factors(60).",
     "def prime_factors",
     ("def prime_factors(n):\n"
      "    factors = []\n"
      "    d = 2\n"
      "    while d * d <= n:\n"
      "        while n % d == 0:\n"
      "            factors.append(d)\n"
      "            n //= d\n"
      "        d += 1\n"
      "    if n > 1:\n"
      "        factors.append(n)\n"
      "    return factors\n"
      "print(prime_factors(60))\n")),

    ("gcd_list",
     "Write a function gcd_list(nums) that returns the greatest common divisor of a list of integers. Print gcd_list([12, 18, 24]).",
     "def gcd_list",
     ("from math import gcd\n"
      "from functools import reduce\n"
      "def gcd_list(nums):\n"
      "    return reduce(gcd, nums)\n"
      "print(gcd_list([12, 18, 24]))\n")),

    ("is_subsequence",
     "Write a function is_subsequence(s, t) that returns True if s is a subsequence of t (characters in order, not necessarily contiguous). Print is_subsequence('abc', 'ahbgdc').",
     "def is_subsequence",
     ("def is_subsequence(s, t):\n"
      "    it = iter(t)\n"
      "    return all(ch in it for ch in s)\n"
      "print(is_subsequence('abc', 'ahbgdc'))\n")),

    # ----------------------------------------------------------------------
    # Batch 2: more tasks for broader coverage
    # ----------------------------------------------------------------------

    ("zip_dict",
     "Write a function zip_dict(keys, values) that creates a dict from two lists. Print zip_dict(['a', 'b', 'c'], [1, 2, 3]).",
     "def zip_dict",
     "def zip_dict(keys, values):\n    return dict(zip(keys, values))\nprint(zip_dict(['a', 'b', 'c'], [1, 2, 3]))\n"),

    ("invert_dict",
     "Write a function invert_dict(d) that swaps keys and values of a dict. Print invert_dict({'a': 1, 'b': 2, 'c': 3}).",
     "def invert_dict",
     "def invert_dict(d):\n    return {v: k for k, v in d.items()}\nprint(invert_dict({'a': 1, 'b': 2, 'c': 3}))\n"),

    ("power_set",
     "Write a function power_set(s) that returns all subsets of a list (order doesn't matter within subsets, sort each subset and the outer list). Print power_set([1, 2, 3]).",
     "def power_set",
     ("from itertools import combinations\n"
      "def power_set(s):\n"
      "    result = []\n"
      "    for r in range(len(s) + 1):\n"
      "        for c in combinations(sorted(s), r):\n"
      "            result.append(list(c))\n"
      "    return result\n"
      "print(power_set([1, 2, 3]))\n")),

    ("intersection",
     "Write a function intersection(a, b) that returns the sorted list of common elements between two lists. Print intersection([1, 2, 3, 4], [3, 4, 5, 6]).",
     "def intersection",
     ("def intersection(a, b):\n"
      "    return sorted(set(a) & set(b))\n"
      "print(intersection([1, 2, 3, 4], [3, 4, 5, 6]))\n")),

    ("symmetric_diff",
     "Write a function symmetric_diff(a, b) that returns the sorted list of elements in either list but not both. Print symmetric_diff([1, 2, 3], [2, 3, 4]).",
     "def symmetric_diff",
     ("def symmetric_diff(a, b):\n"
      "    return sorted(set(a) ^ set(b))\n"
      "print(symmetric_diff([1, 2, 3], [2, 3, 4]))\n")),

    ("max_subarray",
     "Write a function max_subarray(nums) that returns the maximum sum of any contiguous subarray (Kadane's algorithm). Print max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]).",
     "def max_subarray",
     ("def max_subarray(nums):\n"
      "    best = cur = nums[0]\n"
      "    for x in nums[1:]:\n"
      "        cur = max(x, cur + x)\n"
      "        best = max(best, cur)\n"
      "    return best\n"
      "print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))\n")),

    ("spiral_order",
     "Write a function spiral_order(matrix) that returns the elements of a 2D matrix in spiral order. Print spiral_order([[1,2,3],[4,5,6],[7,8,9]]).",
     "def spiral_order",
     ("def spiral_order(matrix):\n"
      "    result = []\n"
      "    while matrix:\n"
      "        result += matrix.pop(0)\n"
      "        if matrix and matrix[0]:\n"
      "            for row in matrix:\n"
      "                result.append(row.pop())\n"
      "        if matrix:\n"
      "            result += matrix.pop()[::-1]\n"
      "        if matrix and matrix[0]:\n"
      "            for row in matrix[::-1]:\n"
      "                result.append(row.pop(0))\n"
      "    return result\n"
      "print(spiral_order([[1,2,3],[4,5,6],[7,8,9]]))\n")),

    ("longest_common_prefix",
     "Write a function longest_common_prefix(strs) that returns the longest common prefix string among a list of strings. Print longest_common_prefix(['flower', 'flow', 'flight']).",
     "def longest_common_prefix",
     ("def longest_common_prefix(strs):\n"
      "    if not strs:\n"
      "        return ''\n"
      "    prefix = strs[0]\n"
      "    for s in strs[1:]:\n"
      "        while not s.startswith(prefix):\n"
      "            prefix = prefix[:-1]\n"
      "        if not prefix:\n"
      "            return ''\n"
      "    return prefix\n"
      "print(longest_common_prefix(['flower', 'flow', 'flight']))\n")),

    ("roman_to_int",
     "Write a function roman_to_int(s) that converts a Roman numeral string to an integer. Print roman_to_int('MCMXCIV').",
     "def roman_to_int",
     ("def roman_to_int(s):\n"
      "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
      "    total = 0\n"
      "    for i in range(len(s)):\n"
      "        if i + 1 < len(s) and vals[s[i]] < vals[s[i+1]]:\n"
      "            total -= vals[s[i]]\n"
      "        else:\n"
      "            total += vals[s[i]]\n"
      "    return total\n"
      "print(roman_to_int('MCMXCIV'))\n")),

    ("valid_parentheses",
     "Write a function generate_parens(n) that returns all combinations of n pairs of well-formed parentheses, sorted lexicographically. Print generate_parens(3).",
     "def generate_parens",
     ("def generate_parens(n):\n"
      "    result = []\n"
      "    def bt(s, o, c):\n"
      "        if len(s) == 2 * n:\n"
      "            result.append(s)\n"
      "            return\n"
      "        if o < n:\n"
      "            bt(s + '(', o + 1, c)\n"
      "        if c < o:\n"
      "            bt(s + ')', o, c + 1)\n"
      "    bt('', 0, 0)\n"
      "    return result\n"
      "print(generate_parens(3))\n")),

    ("group_anagrams",
     "Write a function group_anagrams(words) that groups a list of words by anagram. Return a sorted list of sorted groups. Print group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']).",
     "def group_anagrams",
     ("from collections import defaultdict\n"
      "def group_anagrams(words):\n"
      "    groups = defaultdict(list)\n"
      "    for w in words:\n"
      "        groups[''.join(sorted(w))].append(w)\n"
      "    return sorted(sorted(g) for g in groups.values())\n"
      "print(group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']))\n")),

    ("compress_string",
     "Write a function compress(s) that performs basic string compression: 'aabcccccaaa' -> 'a2b1c5a3'. If compressed is not shorter, return original. Print compress('aabcccccaaa').",
     "def compress",
     ("def compress(s):\n"
      "    if not s:\n"
      "        return s\n"
      "    parts = []\n"
      "    count = 1\n"
      "    for i in range(1, len(s)):\n"
      "        if s[i] == s[i-1]:\n"
      "            count += 1\n"
      "        else:\n"
      "            parts.append(f'{s[i-1]}{count}')\n"
      "            count = 1\n"
      "    parts.append(f'{s[-1]}{count}')\n"
      "    comp = ''.join(parts)\n"
      "    return comp if len(comp) < len(s) else s\n"
      "print(compress('aabcccccaaa'))\n")),

    ("deep_flatten_dict",
     "Write a function flatten_dict(d, sep='.') that flattens a nested dict with dot-separated keys. Print flatten_dict({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}).",
     "def flatten_dict",
     ("def flatten_dict(d, sep='.', prefix=''):\n"
      "    out = {}\n"
      "    for k, v in d.items():\n"
      "        key = f'{prefix}{sep}{k}' if prefix else k\n"
      "        if isinstance(v, dict):\n"
      "            out.update(flatten_dict(v, sep, key))\n"
      "        else:\n"
      "            out[key] = v\n"
      "    return out\n"
      "print(flatten_dict({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}))\n")),

    ("permutations_list",
     "Write a function permutations(lst) that returns all permutations of a list, sorted. Print permutations([1, 2, 3]).",
     "def permutations",
     ("from itertools import permutations as _perms\n"
      "def permutations(lst):\n"
      "    return sorted(list(p) for p in _perms(lst))\n"
      "print(permutations([1, 2, 3]))\n")),

    ("longest_increasing",
     "Write a function lis_length(nums) that returns the length of the longest strictly increasing subsequence. Print lis_length([10, 9, 2, 5, 3, 7, 101, 18]).",
     "def lis_length",
     ("from bisect import bisect_left\n"
      "def lis_length(nums):\n"
      "    tails = []\n"
      "    for x in nums:\n"
      "        pos = bisect_left(tails, x)\n"
      "        if pos == len(tails):\n"
      "            tails.append(x)\n"
      "        else:\n"
      "            tails[pos] = x\n"
      "    return len(tails)\n"
      "print(lis_length([10, 9, 2, 5, 3, 7, 101, 18]))\n")),

    ("zigzag_string",
     "Write a function zigzag(s, n) that converts a string to zigzag pattern with n rows and reads it line by line. Print zigzag('PAYPALISHIRING', 3).",
     "def zigzag",
     ("def zigzag(s, n):\n"
      "    if n <= 1:\n"
      "        return s\n"
      "    rows = [''] * n\n"
      "    row, step = 0, 1\n"
      "    for ch in s:\n"
      "        rows[row] += ch\n"
      "        if row == 0:\n"
      "            step = 1\n"
      "        elif row == n - 1:\n"
      "            step = -1\n"
      "        row += step\n"
      "    return ''.join(rows)\n"
      "print(zigzag('PAYPALISHIRING', 3))\n")),

    ("atoi",
     "Write a function my_atoi(s) that converts a string to a 32-bit signed integer (like C atoi). Handle leading whitespace, optional +/- sign, and stop at the first non-digit. Clamp to [-2^31, 2^31-1]. Print my_atoi('   -42abc').",
     "def my_atoi",
     ("def my_atoi(s):\n"
      "    s = s.lstrip()\n"
      "    if not s:\n"
      "        return 0\n"
      "    sign = 1\n"
      "    i = 0\n"
      "    if s[0] in '+-':\n"
      "        sign = -1 if s[0] == '-' else 1\n"
      "        i = 1\n"
      "    num = 0\n"
      "    while i < len(s) and s[i].isdigit():\n"
      "        num = num * 10 + int(s[i])\n"
      "        i += 1\n"
      "    num *= sign\n"
      "    return max(-(2**31), min(2**31 - 1, num))\n"
      "print(my_atoi('   -42abc'))\n")),

    ("merge_intervals",
     "Write a function merge_intervals(intervals) that merges overlapping intervals. Print merge_intervals([[1,3],[2,6],[8,10],[15,18]]).",
     "def merge_intervals",
     ("def merge_intervals(intervals):\n"
      "    intervals.sort()\n"
      "    merged = [intervals[0]]\n"
      "    for s, e in intervals[1:]:\n"
      "        if s <= merged[-1][1]:\n"
      "            merged[-1][1] = max(merged[-1][1], e)\n"
      "        else:\n"
      "            merged.append([s, e])\n"
      "    return merged\n"
      "print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))\n")),

    ("longest_palindrome_sub",
     "Write a function longest_palindrome(s) that returns the longest palindromic substring. Print longest_palindrome('babad').",
     "def longest_palindrome",
     ("def longest_palindrome(s):\n"
      "    best = ''\n"
      "    for i in range(len(s)):\n"
      "        for lo, hi in [(i, i), (i, i+1)]:\n"
      "            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:\n"
      "                lo -= 1\n"
      "                hi += 1\n"
      "            if hi - lo - 1 > len(best):\n"
      "                best = s[lo+1:hi]\n"
      "    return best\n"
      "print(longest_palindrome('babad'))\n")),

    ("count_islands",
     "Write a function count_islands(grid) that counts the number of islands (connected 1s) in a 2D grid. Print count_islands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]).",
     "def count_islands",
     ("def count_islands(grid):\n"
      "    if not grid:\n"
      "        return 0\n"
      "    rows, cols = len(grid), len(grid[0])\n"
      "    count = 0\n"
      "    def dfs(r, c):\n"
      "        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':\n"
      "            return\n"
      "        grid[r][c] = '0'\n"
      "        dfs(r+1,c); dfs(r-1,c); dfs(r,c+1); dfs(r,c-1)\n"
      "    for r in range(rows):\n"
      "        for c in range(cols):\n"
      "            if grid[r][c] == '1':\n"
      "                count += 1\n"
      "                dfs(r, c)\n"
      "    return count\n"
      "print(count_islands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]))\n")),

    ("eval_rpn",
     "Write a function eval_rpn(tokens) that evaluates a reverse Polish notation expression. Print eval_rpn(['2','1','+','3','*']).",
     "def eval_rpn",
     ("def eval_rpn(tokens):\n"
      "    stack = []\n"
      "    for t in tokens:\n"
      "        if t in '+-*/':\n"
      "            b, a = stack.pop(), stack.pop()\n"
      "            if t == '+': stack.append(a + b)\n"
      "            elif t == '-': stack.append(a - b)\n"
      "            elif t == '*': stack.append(a * b)\n"
      "            else: stack.append(int(a / b))\n"
      "        else:\n"
      "            stack.append(int(t))\n"
      "    return stack[0]\n"
      "print(eval_rpn(['2','1','+','3','*']))\n")),

    ("encode_decode",
     "Write functions encode(strs) that encodes a list of strings into a single string and decode(s) that reverses it. Print decode(encode(['hello', 'world'])).",
     "def encode",
     ("def encode(strs):\n"
      "    return ''.join(f'{len(s)}#{s}' for s in strs)\n"
      "def decode(s):\n"
      "    result, i = [], 0\n"
      "    while i < len(s):\n"
      "        j = s.index('#', i)\n"
      "        length = int(s[i:j])\n"
      "        result.append(s[j+1:j+1+length])\n"
      "        i = j + 1 + length\n"
      "    return result\n"
      "print(decode(encode(['hello', 'world'])))\n")),

    ("product_except_self",
     "Write a function product_except_self(nums) that returns a list where each element is the product of all other elements, without using division. Print product_except_self([1, 2, 3, 4]).",
     "def product_except_self",
     ("def product_except_self(nums):\n"
      "    n = len(nums)\n"
      "    out = [1] * n\n"
      "    left = 1\n"
      "    for i in range(n):\n"
      "        out[i] = left\n"
      "        left *= nums[i]\n"
      "    right = 1\n"
      "    for i in range(n - 1, -1, -1):\n"
      "        out[i] *= right\n"
      "        right *= nums[i]\n"
      "    return out\n"
      "print(product_except_self([1, 2, 3, 4]))\n")),

    ("min_window_substring",
     "Write a function min_window(s, t) that returns the minimum window substring of s that contains all characters in t. Print min_window('ADOBECODEBANC', 'ABC').",
     "def min_window",
     ("from collections import Counter\n"
      "def min_window(s, t):\n"
      "    need = Counter(t)\n"
      "    missing = len(t)\n"
      "    best = ''\n"
      "    j = 0\n"
      "    for i, ch in enumerate(s):\n"
      "        if need[ch] > 0:\n"
      "            missing -= 1\n"
      "        need[ch] -= 1\n"
      "        while missing == 0:\n"
      "            window = s[j:i+1]\n"
      "            if not best or len(window) < len(best):\n"
      "                best = window\n"
      "            need[s[j]] += 1\n"
      "            if need[s[j]] > 0:\n"
      "                missing += 1\n"
      "            j += 1\n"
      "    return best\n"
      "print(min_window('ADOBECODEBANC', 'ABC'))\n")),

    ("lru_cache_impl",
     "Implement an LRUCache class with get(key) and put(key, value) methods (capacity 2). Do: put(1,1), put(2,2), print get(1), put(3,3), print get(2), put(4,4), print get(1), print get(3), print get(4).",
     "class LRUCache",
     ("from collections import OrderedDict\n"
      "class LRUCache:\n"
      "    def __init__(self, capacity):\n"
      "        self.cap = capacity\n"
      "        self.cache = OrderedDict()\n"
      "    def get(self, key):\n"
      "        if key not in self.cache:\n"
      "            return -1\n"
      "        self.cache.move_to_end(key)\n"
      "        return self.cache[key]\n"
      "    def put(self, key, value):\n"
      "        if key in self.cache:\n"
      "            self.cache.move_to_end(key)\n"
      "        self.cache[key] = value\n"
      "        if len(self.cache) > self.cap:\n"
      "            self.cache.popitem(last=False)\n"
      "c = LRUCache(2)\n"
      "c.put(1,1); c.put(2,2)\n"
      "print(c.get(1))\n"
      "c.put(3,3)\n"
      "print(c.get(2))\n"
      "c.put(4,4)\n"
      "print(c.get(1))\n"
      "print(c.get(3))\n"
      "print(c.get(4))\n")),

    ("trie_search",
     "Implement a Trie class with insert(word) and search(word) methods. Insert 'apple' and 'app', then print search('apple'), search('app'), search('ap').",
     "class Trie",
     ("class Trie:\n"
      "    def __init__(self):\n"
      "        self.children = {}\n"
      "        self.end = False\n"
      "    def insert(self, word):\n"
      "        node = self\n"
      "        for ch in word:\n"
      "            if ch not in node.children:\n"
      "                node.children[ch] = Trie()\n"
      "            node = node.children[ch]\n"
      "        node.end = True\n"
      "    def search(self, word):\n"
      "        node = self\n"
      "        for ch in word:\n"
      "            if ch not in node.children:\n"
      "                return False\n"
      "            node = node.children[ch]\n"
      "        return node.end\n"
      "t = Trie()\n"
      "t.insert('apple')\n"
      "t.insert('app')\n"
      "print(t.search('apple'))\n"
      "print(t.search('app'))\n"
      "print(t.search('ap'))\n")),

    ("top_k_frequent",
     "Write a function top_k_frequent(nums, k) that returns the k most frequent elements sorted by frequency descending. Print top_k_frequent([1,1,1,2,2,3], 2).",
     "def top_k_frequent",
     ("from collections import Counter\n"
      "def top_k_frequent(nums, k):\n"
      "    return [x for x, _ in Counter(nums).most_common(k)]\n"
      "print(top_k_frequent([1,1,1,2,2,3], 2))\n")),

    ("valid_sudoku_row",
     "Write a function is_valid_row(row) that returns True if a 9-element list has no duplicate digits 1-9 (zeros are blanks). Print is_valid_row([5,3,0,0,7,0,0,0,0]) and is_valid_row([5,3,0,0,7,0,5,0,0]).",
     "def is_valid_row",
     ("def is_valid_row(row):\n"
      "    digits = [x for x in row if x != 0]\n"
      "    return len(digits) == len(set(digits))\n"
      "print(is_valid_row([5,3,0,0,7,0,0,0,0]))\n"
      "print(is_valid_row([5,3,0,0,7,0,5,0,0]))\n")),
]


# ---------------------------------------------------------------------------
# Verification + file IO
# ---------------------------------------------------------------------------

def run_capture(python_bin: str, code: str, timeout: float) -> str:
    """Run `code` as a script and return its stdout. Raises on failure."""
    with tempfile.TemporaryDirectory(prefix="codeeval_") as tmp:
        sol = Path(tmp) / "sol.py"
        sol.write_text(code)
        proc = subprocess.run(
            [python_bin, str(sol)], capture_output=True, text=True, timeout=timeout
        )
    if proc.returncode != 0:
        raise RuntimeError(f"reference solution exited {proc.returncode}:\n{proc.stderr}")
    return proc.stdout


def load_existing(data_path: Path) -> tuple[list[dict], set[str], int]:
    """Return (records, set of existing `user` prompts, max code_NNN index)."""
    records: list[dict] = []
    users: set[str] = set()
    max_idx = 0
    if data_path.exists():
        for line in data_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(rec)
            users.add(rec.get("user", ""))
            m = _ID_RE.fullmatch(rec.get("id", "")) or _ID_RE.match(rec.get("id", ""))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return records, users, max_idx


def build_records(python_bin: str, timeout: float, existing_users: set[str],
                  start_idx: int) -> list[dict]:
    """Verify every not-yet-present task and return new records (with ids)."""
    new_records: list[dict] = []
    idx = start_idx
    for slug, user, must, sol in TASKS:
        if user in existing_users:
            print(f"  skip (already present)        {slug}")
            continue
        out1 = run_capture(python_bin, sol, timeout)
        out2 = run_capture(python_bin, sol, timeout)
        if out1 != out2:
            raise RuntimeError(f"NON-DETERMINISTIC reference solution: {slug!r}")
        rec = {
            "id": f"code_{idx:03d}",
            "user": user,
            "checks": {"must_include_any": [[must]]},
            "code_exec": {"expected_stdout": out1, "timeout_s": int(timeout)},
        }
        new_records.append(rec)
        idx += 1
        print(f"  {rec['id']}  {slug:<22}  expected_stdout={out1!r}")
    return new_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default=str(_DEFAULT_DATA),
                   help="Eval JSONL to append to (default: data/python_code_eval.jsonl).")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used to run reference solutions (default: this one).")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Per-solution timeout (also stored as code_exec.timeout_s).")
    p.add_argument("--dry-run", action="store_true",
                   help="Verify and print new tasks, but do not write the file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    existing, existing_users, max_idx = load_existing(data_path)
    print(f"Existing: {len(existing)} tasks (max id index {max_idx}) in {data_path}\n")

    new_records = build_records(args.python, args.timeout, existing_users, max_idx + 1)

    if not new_records:
        print("\nNo new tasks to add (all TASKS already present).")
        return

    if args.dry_run:
        print(f"\n[dry-run] {len(new_records)} new tasks verified; nothing written.")
        return

    with open(data_path, "a") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")

    # Re-read and assert integrity (fail loud on duplicate ids).
    final, _, _ = load_existing(data_path)
    ids = [r["id"] for r in final]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise RuntimeError(f"duplicate ids after append: {dupes}")
    print(f"\nAppended {len(new_records)} tasks. Total now {len(final)} "
          f"({ids[0]} -> {ids[-1]}).")


if __name__ == "__main__":
    main()
