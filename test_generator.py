import unittest

from app import (
    extract_sections,
    validate_question_paper,
    count_numbered_questions,
    declared_total_marks,
)


class TestQuestionPaperGenerator(unittest.TestCase):
    def test_separate_student_paper_and_answer_key(self):
        text = """STUDENT QUESTION PAPER

Total Marks: 20
1. What is a variable? [Bloom: Remember] [2 marks]

TEACHER ANSWER KEY

1. A variable stores a value.
"""
        student, answer_key = extract_sections(text)
        self.assertIn("What is a variable?", student)
        self.assertNotIn("A variable stores a value.", student)
        self.assertIn("A variable stores a value.", answer_key)

    def test_missing_answer_key(self):
        student, answer_key = extract_sections("1. What is a variable?")
        self.assertTrue(student)
        self.assertEqual(answer_key, "")

    def test_empty_question_paper(self):
        issues = validate_question_paper("", "", 5, 20)
        self.assertTrue(any("empty" in issue.lower() for issue in issues))

    def test_wrong_question_count(self):
        paper = """Total Marks: 20

1. Define a variable. [Bloom: Remember] [10 marks]
2. Explain a loop. [Bloom: Understand] [10 marks]
"""
        issues = validate_question_paper(paper, "1. Answer\n2. Answer", 5, 20)
        self.assertTrue(any("Expected 5 questions" in issue for issue in issues))

    def test_missing_bloom_label(self):
        paper = """Total Marks: 20

1. Define a variable. [2 marks]
2. Explain a loop. [18 marks]
"""
        issues = validate_question_paper(paper, "1. Answers\n2. Answers", 2, 20)
        self.assertTrue(any("Bloom" in issue for issue in issues))

    def test_question_count_helper(self):
        paper = "1. First question\n2) Second question\nSome extra text"
        self.assertEqual(count_numbered_questions(paper), 2)

    def test_total_marks_helper(self):
        self.assertEqual(declared_total_marks("Total Marks: 50"), 50)
        self.assertIsNone(declared_total_marks("No marks mentioned"))

    def test_wrong_declared_marks(self):
        paper = """Total Marks: 25

1. Define a variable. [Bloom: Remember] [25 marks]
"""
        issues = validate_question_paper(paper, "1. Answer", 1, 20)
        self.assertTrue(any("paper declares 25" in issue.lower() for issue in issues))

    def test_answer_key_marking_scheme_mismatch(self):
        paper = """Total Marks: 4

1. Write a C program. [Bloom: Apply] [4 marks]
"""
        key = """1. Answer: sample program
Marking points: 1 for structure, 1 for input, 1 for calculation,
1 for file handling, 1 for correctness (total 5, but only 4 marks allocated).
"""
        issues = validate_question_paper(paper, key, 1, 4)
        self.assertTrue(any("marking scheme totals 5" in issue.lower() for issue in issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
