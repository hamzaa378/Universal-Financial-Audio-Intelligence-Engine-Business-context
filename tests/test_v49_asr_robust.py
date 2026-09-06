import unittest
from nlp.pii import detect_pii, mask_pii


class V49ASRRobustTests(unittest.TestCase):
    def hits(self, text):
        return detect_pii(text)

    def kinds(self, text):
        return [x['type'] for x in self.hits(text)]

    def assert_one(self, text, kind):
        hits=self.hits(text)
        self.assertEqual([x['type'] for x in hits],[kind],text)
        self.assertNotEqual(mask_pii(text,hits),text)
        return hits[0]

    def test_asr_corrupted_phone_with_strong_ownership(self):
        hit=self.assert_one('You can also contact me at plus 911-23456789.','PHONE')
        self.assertTrue(hit.get('raw_fallback'))

    def test_mixed_spoken_written_email(self):
        for text in (
            'My email address is arov.com at example.com.',
            'My alternate email is arove.private at mail.co.in.',
            'Send the confirmation to sara.private at mail.co.in.',
            'Email me at sara dot khan at mail dot co dot in.',
        ):
            self.assert_one(text,'EMAIL')

    def test_mixed_pan_phonetic_and_written_digits(self):
        hit=self.assert_one('My PAN number is Alpha Bravo Charlie Delta Echo 1234 Foxtrot.','PAN')
        self.assertEqual(hit.get('canonical'),'ABCDE1234F')

    def test_ifsc_raw_privacy_fallback(self):
        hit=self.assert_one('My bank IFSC is Hotel Delta Foxtrott ID, Charlie 00001234.','IFSC')
        self.assertTrue(hit.get('raw_fallback'))

    def test_upi_mixed_and_payment_language(self):
        for text in (
            'My UPI address is rf.con at OKSBI.',
            'Transfer the refund to sara.private at oksbi.',
            'Pay me at sara dot khan at oksbi.',
        ):
            self.assert_one(text,'UPI')

    def test_damaged_card_masks_only_with_ownership(self):
        self.assert_one('My credit card number is 4111-1111-111.','CARD')
        self.assertFalse(self.hits('The test card number is 4111-1111-111.'))

    def test_asr_pincode_insert_delete_tolerance(self):
        self.assert_one('My PIN code is 4111045.','PINCODE')
        self.assertFalse(self.hits('The postal code 4111045 covers part of Pune.'))

    def test_passport_case_insensitive_nato(self):
        hit=self.assert_one('My passport is PAPA 1234567.','PASSPORT')
        self.assertEqual(hit.get('canonical'),'P1234567')

    def test_fuzzy_state_driving_license(self):
        self.assert_one('My driving license number is Carnitaka0120210012345.','DRIVING_LICENSE')

    def test_explanatory_sentences_are_not_values(self):
        negatives=(
            'A residential address may contain a house number, road, city and PIN code.',
            'A UPI ID usually contains an at sign.',
            'A mobile number may contain ten digits such as 9876543210.',
            'An IFSC identifies a bank branch.',
            'A passport number may contain letters and digits.',
        )
        for text in negatives:
            self.assertFalse(self.hits(text),text)

    def test_public_phone_lines_are_not_personal(self):
        for text in (
            'The customer support helpline is 911-2456789.',
            'The support hotline is 98765-43210.',
        ):
            self.assertFalse(self.hits(text),text)

    def test_broader_natural_ownership_phrases(self):
        cases=(
            ('I prefer to be called Aisha Noor.','NAME'),
            ('Please address me as Sara Khan.','NAME'),
            ('The name on my account is Sara Khan.','NAME'),
            ('The best number to reach me is 9988776655.','PHONE'),
            ('SMS me on 9876543210.','PHONE'),
            ('My mailing address is 10 Green Road, Pune 411001.','ADDRESS'),
            ('You can find me at Flat 2, Lake Road, Pune 411001.','ADDRESS'),
            ('Mail the correspondence to Flat 3, Lake Road, Pune 411001.','ADDRESS'),
        )
        for text,kind in cases:
            self.assert_one(text,kind)


if __name__ == '__main__':
    unittest.main(verbosity=2)
