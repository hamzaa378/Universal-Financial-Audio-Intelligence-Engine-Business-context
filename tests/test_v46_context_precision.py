import unittest

from nlp.pii import detect_pii, mask_pii
from nlp.privacy import protect_text


class V46ContextPrecisionTests(unittest.TestCase):
    def kinds(self, text):
        return {x["type"] for x in detect_pii(text)}

    def test_documentation_examples_are_not_pii(self):
        negatives = [
            "The documentation example uses arjun.mehta94@gmail.com as a sample email address.",
            "The product serial number is BCDPM4321K.",
            "The configuration file contains the literal text ICIC0001234 as a test value.",
            "The software test documentation uses 4111 1111 1111 1111 as a standard test card number.",
            "The documentation states that rohit@oksbi is the example VPA used throughout this manual.",
        ]
        for text in negatives:
            self.assertFalse(detect_pii(text), text)

    def test_same_literal_gets_occurrence_specific_semantics(self):
        text = "My DOB is 21 January 2002, while the application deadline was also 21 January 2002."
        hits = [x for x in detect_pii(text) if x["type"] == "DOB"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["value"], "21 January 2002")
        masked = mask_pii(text, hits)
        self.assertEqual(masked.count("[DOB REDACTED]"), 1)
        self.assertTrue(masked.endswith("21 January 2002."))

    def test_role_does_not_leak_across_new_field(self):
        account_text = "My account number is 876543210987 and the transaction ID is 876543210987."
        accounts = [x for x in detect_pii(account_text) if x["type"] == "ACCOUNT_NUMBER"]
        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0]["start"], account_text.index("876543210987"))

        phone_text = "My phone number is 9123456789 and the complaint ID is 9123456789."
        phones = [x for x in detect_pii(phone_text) if x["type"] == "PHONE"]
        self.assertEqual(len(phones), 1)
        self.assertEqual(phones[0]["start"], phone_text.index("9123456789"))

    def test_implicit_phone_ownership(self):
        for text in (
            "The customer told me his number is 9988776655.",
            "Her number is 8877665544.",
            "Reach Rahul on 9988776655.",
            "Please save 7766554433 for future contact.",
        ):
            self.assertIn("PHONE", self.kinds(text), text)
        self.assertNotIn("PHONE", self.kinds("The number 6655443322 appears in the report."))
        self.assertNotIn("PHONE", self.kinds("Rahul scored 9988776655 points in the synthetic benchmark."))

    def test_delivery_address_without_address_label(self):
        text = "Send the package to Flat 9B, Sunrise Apartments, Baner Road, Pune 411045."
        hits = detect_pii(text)
        address = next(x for x in hits if x["type"] == "ADDRESS")
        self.assertEqual(address["value"], "Flat 9B, Sunrise Apartments, Baner Road, Pune 411045")

    def test_address_stops_before_contrastive_public_clause(self):
        text = "My permanent address is House 18, Lake Road, Bhopal 462003, but Lake Road itself is a public street name."
        hits = detect_pii(text)
        address = next(x for x in hits if x["type"] == "ADDRESS")
        self.assertEqual(address["value"], "House 18, Lake Road, Bhopal 462003")
        masked = mask_pii(text, hits)
        self.assertIn("but Lake Road itself is a public street name", masked)

    def test_postal_code_does_not_become_whole_address(self):
        text = "The postal code for my address is 110016."
        hits = detect_pii(text)
        self.assertTrue(any(x["type"] == "PINCODE" for x in hits))
        self.assertFalse(any(x["type"] == "ADDRESS" for x in hits))

    def test_address_abbreviation_does_not_end_span(self):
        text = "My address is 17, Lake View Rd., Pune - 411001."
        address = next(x for x in detect_pii(text) if x["type"] == "ADDRESS")
        self.assertEqual(address["value"], "17, Lake View Rd., Pune - 411001")

    def test_self_identification_name(self):
        self.assertIn("NAME", self.kinds("For final verification, I am Neha Kapoor."))
        self.assertNotIn("NAME", self.kinds("I am calling regarding a payment issue."))

    def test_spoken_alphanumeric_pan_and_ifsc(self):
        pan = "My PAN is Alpha Bravo Charlie Delta Echo one two three four Foxtrot."
        pan_hit = next(x for x in detect_pii(pan) if x["type"] == "PAN")
        self.assertEqual(pan_hit.get("canonical"), "ABCDE1234F")

        ifsc = "My bank IFSC is Hotel Delta Foxtrot Charlie zero zero zero one two three four."
        ifsc_hit = next(x for x in detect_pii(ifsc) if x["type"] == "IFSC")
        self.assertEqual(ifsc_hit.get("canonical"), "HDFC0001234")

    def test_training_example_spoken_email_is_not_masked(self):
        text = "The training video says an email can be spoken as name dot surname at domain dot com."
        self.assertFalse(detect_pii(text))

    def test_credentials_are_context_gated(self):
        positive = {
            "My password is winterMoon#83.": "PASSWORD",
            "My username is arjun_m94.": "USERNAME",
            "My API key is sk-test-51KLMN7832xyzPQ098765.": "API_KEY",
            "Please use token ghp_3XyExampleToken1234567890abcDEF for authentication.": "AUTH_TOKEN",
            'OPENAI_API_KEY="sk-proj-51KLMN7832xyzPQ098765"': "API_KEY",
            'GITHUB_TOKEN="ghp_3XyExampleToken1234567890abcDEF"': "AUTH_TOKEN",
        }
        for text, kind in positive.items():
            self.assertIn(kind, self.kinds(text), text)

        negatives = [
            'The source code tutorial contains API_KEY="YOUR_API_KEY_HERE".',
            'The source code tutorial contains OPENAI_API_KEY="YOUR_API_KEY_HERE".',
            "The environment variable name is OPENAI_API_KEY.",
            "The documentation says GitHub tokens often start with ghp_.",
            "The article recommends that a password should contain letters, numbers and symbols.",
            "The UI field is labelled username.",
        ]
        for text in negatives:
            self.assertFalse(detect_pii(text), text)

    def test_operational_ids_are_opt_in(self):
        text = "Transaction reference TXN-88991122 and ticket number 638291."
        safe_default = protect_text(text)["text"]
        self.assertIn("TXN-88991122", safe_default)
        self.assertIn("638291", safe_default)
        safe_opt_in = protect_text(text, financial_ids_enabled=True)["text"]
        self.assertNotIn("TXN-88991122", safe_opt_in)
        self.assertNotIn("638291", safe_opt_in)


if __name__ == "__main__":
    unittest.main(verbosity=2)
