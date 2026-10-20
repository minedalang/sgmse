"""Fixed literal synthetic cases only; no data discovery, audio or model use."""
from decimal import Decimal
import unittest
from annotation_contract import translate_annotation


def call(label, **changes):
    args = dict(source_kind="manual_annotation", source_id="synthetic:unit",
                start_s="0.1000", end_s="0.2500")
    args.update(changes)
    return translate_annotation(label, **args)


class ContractTests(unittest.TestCase):
    def test_substitution_preserves_actual_not_expected(self):
        r = call("K,G,s")
        for key, value in {"raw_label":"K,G,s", "expected_phone":"K",
                           "realized_phone":"G", "raw_perceived_symbol":"G",
                           "annotation_operation":"s", "realized_class":"speech",
                           "identity_uncertain":False, "deviation":False}.items():
            self.assertEqual(r[key], value)

    def test_source_deletion_not_enhancement_deletion(self):
        r = call("K,sil,d")
        self.assertEqual(r["realized_phone"], "sil")
        self.assertEqual(r["expected_phone"], "K")
        self.assertEqual(r["realized_class"], "silence")
        self.assertEqual(r["operation_scope"], "source_pronunciation_not_enhancement")

    def test_source_addition_is_present(self):
        r = call("sil,K,a")
        self.assertEqual((r["expected_phone"], r["realized_phone"], r["realized_class"]),
                         ("sil", "K", "speech"))
        self.assertEqual(r["annotation_operation"], "a")

    def test_error_identity_not_silence(self):
        r = call("K,err,s")
        self.assertEqual(r["raw_perceived_symbol"], "err")
        self.assertIsNone(r["realized_phone"])
        self.assertTrue(r["identity_uncertain"])
        self.assertEqual(r["realized_class"], "uncertain")
        self.assertFalse(r["deviation"])

    def test_star_retained_not_hard_g(self):
        r = call("K,G*,s")
        self.assertEqual(r["raw_perceived_symbol"], "G*")
        self.assertEqual(r["perceived_symbol_normalized"], "G*")
        self.assertIsNone(r["realized_phone"])
        self.assertTrue(r["deviation"])
        self.assertTrue(r["identity_uncertain"])

    def test_single_sources_are_different(self):
        m, f = call("K"), call("K", source_kind="forced_alignment")
        self.assertEqual(m["annotation_basis"], "manual_source_assertion")
        self.assertEqual(f["annotation_basis"], "forced_alignment_estimate")
        for r in (m, f):
            self.assertEqual(r["realized_phone"], "K")
            self.assertIsNone(r["annotation_operation"])
            self.assertFalse(r["hard_identity_eligible"])

    def test_sil_is_source_claim_only(self):
        r = call("sil", start_s="1.20", end_s="1.20")
        self.assertEqual(r["realized_class"], "silence")
        self.assertTrue(r["zero_length"])
        self.assertEqual(r["start_s"], "1.20")
        self.assertIsNone(r["annotation_operation"])

    def test_case_and_raw_fields_preserved(self):
        r = call("k,g,s", source_id=" synthetic:id ")
        self.assertEqual(r["raw_label"], "k,g,s")
        self.assertEqual(r["expected_phone"], "k")
        self.assertEqual(r["realized_phone"], "g")
        self.assertEqual(r["realized_phone_normalized"], "G")
        self.assertEqual(r["source_id"], " synthetic:id ")

    def test_decimal_precision_and_supported_types(self):
        r = call("K", start_s=Decimal("0.123456789012345678901234567890"), end_s="1e0")
        self.assertEqual(r["start_s"], "0.123456789012345678901234567890")
        self.assertEqual(r["end_s"], "1")
        self.assertEqual(r["raw_end_s"], "1e0")
        self.assertEqual(r["start_input_type"], "Decimal")
        self.assertEqual(call("K", start_s=0, end_s=0.25)["end_s"], "0.25")

    def test_all_records_stay_unapproved(self):
        for label in ("K", "sil", "err", "K*", "K,G,s", "K,sil,d", "sil,K,a", "K,err,s", "K,G*,s"):
            with self.subTest(label=label):
                r = call(label)
                self.assertIs(r["real_data_approved"], False)
                self.assertIs(r["hard_identity_eligible"], False)
                self.assertNotIn("waveform_action", r)
                self.assertNotIn("model_unknown", r)

    def test_bad_labels_fail_without_repair(self):
        for label in ("", " ", " K", "K ", "K,,s", ",G,s", "K,G,", "K,G,s,",
                      "K,G", "K,G,x", "K,G,S", "K/G", "G**", "K,G*,s,extra",
                      "K,G,a", "sil,sil,a", "K,G,d", "sil,sil,d", "sil,G,s",
                      "K,sil,s", "K,K,s", "err,G,s", "sil*", "err*", "K, G,s"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError): call(label)

    def test_forced_error_labels_rejected(self):
        for label in ("K,G,s", "K,sil,d", "sil,K,a", "K,err,s"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError): call(label, source_kind="forced_alignment")

    def test_invalid_types_and_sources(self):
        for value in (None, 7, True, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError): call(value)
        for value in (None, [], False):
            with self.assertRaises(TypeError): call("K", source_kind=value)
            with self.assertRaises(TypeError): call("K", source_id=value)
        for value in ("manual", "", "MANUAL_ANNOTATION"):
            with self.assertRaises(ValueError): call("K", source_kind=value)
        for value in ("", "  ", "\n"):
            with self.assertRaises(ValueError): call("K", source_id=value)

    def test_invalid_times(self):
        for value in (True, False, None, [], {}, complex(1, 0)):
            for key in ("start_s", "end_s"):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(TypeError): call("K", **{key:value})
        for value in ("NaN", "sNaN", "Infinity", "-Infinity", float("nan"),
                      float("inf"), Decimal("NaN"), "-0.01", -1, "", " ", " 0.1", "bad"):
            for key in ("start_s", "end_s"):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError): call("K", **{key:value})
        with self.assertRaises(ValueError): call("K", start_s="0.3", end_s="0.2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
