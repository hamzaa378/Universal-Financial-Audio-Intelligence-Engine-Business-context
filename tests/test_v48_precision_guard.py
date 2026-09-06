import unittest

from nlp.pii import detect_pii, mask_pii


class V48PrecisionGuardTests(unittest.TestCase):
    def hits(self, text):
        return detect_pii(text)

    def kinds(self, text):
        return [x["type"] for x in self.hits(text)]

    def test_public_geo_pincode_is_not_personal_pii(self):
        text = "The postal code 411001 covers part of Pune."
        self.assertNotIn("PINCODE", self.kinds(text))
        self.assertEqual(mask_pii(text, self.hits(text)), text)

    def test_personal_pincode_still_masks(self):
        for text in (
            "My PIN code is 411045.",
            "The postal code for my address is 110016.",
        ):
            self.assertIn("PINCODE", self.kinds(text), text)

    def test_address_stops_before_alternate_phone_field(self):
        text = "My address is 15 Rose Garden Road, Chennai 600028, and my alternate phone number is 9988776655."
        hits = self.hits(text)
        self.assertEqual([x["type"] for x in hits], ["ADDRESS", "PHONE"])
        self.assertEqual(
            mask_pii(text, hits),
            "My address is [ADDRESS REDACTED], and my alternate phone number is [PHONE REDACTED].",
        )

    def test_address_stops_before_secondary_email_field(self):
        text = "My address is Flat 8, Lake Road, Pune 411001, and my secondary email is sara@example.com."
        hits = self.hits(text)
        self.assertEqual([x["type"] for x in hits], ["ADDRESS", "EMAIL"])
        self.assertIn(", and my secondary email is [EMAIL REDACTED]", mask_pii(text, hits))

    def test_owned_card_last4_masks(self):
        text = "My card ends with 1111."
        hits = self.hits(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["type"], "CARD")
        self.assertTrue(hits[0].get("partial_identifier"))
        self.assertEqual(mask_pii(text, hits), "My card ends with [CARD REDACTED].")
        # Partial-mask mode must not reveal the only four sensitive digits again.
        self.assertEqual(mask_pii(text, hits, mode="partial"), "My card ends with [CARD REDACTED].")

    def test_owned_account_last4_masks(self):
        text = "The last four digits of my account are 9012."
        hits = self.hits(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["type"], "ACCOUNT_NUMBER")
        self.assertTrue(hits[0].get("partial_identifier"))
        self.assertEqual(
            mask_pii(text, hits),
            "The last four digits of my account are [ACCOUNT_NUMBER REDACTED].",
        )

    def test_partial_identifier_false_positive_guards(self):
        negatives = (
            "I paid 1111 rupees yesterday.",
            "Reference number 9012 appears on page two.",
            "The test card ends with 1111.",
            "The report account ends with 9012.",
            "There were 1111 transactions today.",
        )
        for text in negatives:
            self.assertFalse(self.hits(text), text)

    def test_v47_multifield_behavior_is_retained(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
