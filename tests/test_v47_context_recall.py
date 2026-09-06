import unittest

from nlp.pii import detect_pii, mask_pii


class V47ContextRecallTests(unittest.TestCase):
    def hits(self, text):
        return detect_pii(text)

    def kinds(self, text):
        return [x["type"] for x in self.hits(text)]

    def test_long_multifield_clause_segmentation(self):
        text = (
            "Final combined sentence: my name is Sara Mohammed Khan, DOB 21 January 2002, "
            "phone plus nine one nine eight seven six five four three two one zero, "
            "email sara dot khan at example dot com, account number zero zero one two three four "
            "five six seven eight nine zero one two, IFSC HDFC zero zero zero one two three four, "
            "and address Flat number nine B, Sunrise Apartments, Baner Road, Pune four one one zero four five."
        )
        self.assertEqual(
            self.kinds(text),
            ["NAME", "DOB", "PHONE", "EMAIL", "ACCOUNT_NUMBER", "IFSC", "ADDRESS"],
        )
        masked = mask_pii(text, self.hits(text))
        for kind in ("NAME","DOB","PHONE","EMAIL","ACCOUNT_NUMBER","IFSC","ADDRESS"):
            self.assertIn(f"[{kind} REDACTED]", masked)

    def test_one_sentence_expected_type_state(self):
        cases = {
            "Please enter your Aadhaar number. Mine is 4567 8901 2345.": "AADHAAR",
            "Type your mobile number below. Mine is 9098765432.": "PHONE",
            "An OTP will be sent shortly. I received 843291.": "OTP",
            "The CVV is located on the back of a payment card. Mine is 617.": "CVV",
        }
        for text, kind in cases.items():
            self.assertIn(kind, self.kinds(text), text)

    def test_expected_state_requires_response_ownership(self):
        self.assertNotIn(
            "AADHAAR",
            self.kinds("Please enter your Aadhaar number. The invoice number is 4567 8901 2345."),
        )
        # TTL is one sentence only.
        self.assertNotIn(
            "AADHAAR",
            self.kinds("Please enter your Aadhaar number. Let me check. Mine is 4567 8901 2345."),
        )

    def test_ownership_beats_ambiguous_format(self):
        text = "My account is 4111 1111 1111 1111."
        hits = self.hits(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["type"], "ACCOUNT_NUMBER")
        self.assertGreaterEqual(hits[0].get("ownership_strength", 0), 4)

    def test_spoken_dob(self):
        hit = next(x for x in self.hits("My date of birth is twenty-first January two thousand two.") if x["type"] == "DOB")
        self.assertEqual(hit.get("canonical"), "2002-01-21")

    def test_spoken_upi(self):
        hit = next(x for x in self.hits("My UPI address is sara dot khan at oksbi.") if x["type"] == "UPI")
        self.assertEqual(hit.get("canonical"), "sara.khan@oksbi")
        self.assertFalse(self.hits("The documentation example says user dot name at oksbi."))

    def test_spoken_passport(self):
        hit = next(x for x in self.hits("My passport is Papa one two three four five six seven.") if x["type"] == "PASSPORT")
        self.assertEqual(hit.get("canonical"), "P1234567")

    def test_spoken_driving_licence(self):
        hit = next(
            x for x in self.hits(
                "My driving licence number is Karnataka zero one, two zero two one, zero zero one two three four five."
            ) if x["type"] == "DRIVING_LICENSE"
        )
        self.assertEqual(hit.get("canonical"), "KA0120210012345")

    def test_full_and_registered_name(self):
        self.assertIn("NAME", self.kinds("My full name is Sara Mohammed Khan."))
        self.assertIn("NAME", self.kinds("My registered name is Aisha Noor."))
        self.assertNotIn("NAME", self.kinds("Raj Kumar is the name of the support agent shown on the public help page."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
