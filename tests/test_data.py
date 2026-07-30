import io
import json
import unittest
import zipfile

from qubolens.data import (
    inspect_tabular_upload,
    load_csv_dataset,
    load_tabular_dataset,
    make_demo,
    parse_tabular_upload,
)


class DatasetTests(unittest.TestCase):
    def test_demo_is_deterministic(self):
        first = make_demo("edge-failure")
        second = make_demo("edge-failure")
        self.assertEqual(first.rows[:3], second.rows[:3])
        self.assertEqual(first.target[:20], second.target[:20])
        self.assertEqual(first.n_features, 18)

    def test_csv_loader_encodes_categories_and_imputes(self):
        lines = ["age,region,signal,outcome"]
        for index in range(36):
            age = "" if index == 3 else str(20 + index)
            region = ("north", "south", "west")[index % 3]
            signal = str(index % 7)
            outcome = "yes" if index % 2 else "no"
            lines.append(f"{age},{region},{signal},{outcome}")
        dataset = load_csv_dataset(
            "\n".join(lines),
            target_name="outcome",
            task="classification",
        )
        self.assertEqual(dataset.n_samples, 36)
        self.assertEqual(dataset.n_features, 5)
        self.assertEqual(set(dataset.target), {0.0, 1.0})
        self.assertTrue(any("Expanded region" in note for note in dataset.notes))
        self.assertTrue(any("region =" in name for name in dataset.feature_names))

    def test_csv_requires_enough_rows(self):
        with self.assertRaisesRegex(ValueError, "at least 30"):
            load_csv_dataset(
                "x,z,y\n1,2,0\n2,4,1\n",
                target_name="y",
                task="classification",
            )

    def test_tsv_and_jsonl_support_text_columns(self):
        tsv_lines = ["age\treview\toutcome"]
        jsonl_rows = []
        for index in range(40):
            outcome = "yes" if index % 2 else "no"
            review = (
                f"Device report {index} has repeated thermal warning words"
                if outcome == "yes"
                else f"Device report {index} shows stable normal operation"
            )
            tsv_lines.append(f"{20 + index}\t{review}\t{outcome}")
            jsonl_rows.append({"age": 20 + index, "review": review, "outcome": outcome})

        tsv = load_tabular_dataset(
            "\n".join(tsv_lines).encode(),
            filename="reports.tsv",
            target_name="outcome",
        )
        jsonl = load_tabular_dataset(
            "\n".join(json.dumps(row) for row in jsonl_rows).encode(),
            filename="reports.jsonl",
            target_name="outcome",
        )

        self.assertEqual(tsv.task, "classification")
        self.assertEqual(jsonl.task, "classification")
        self.assertTrue(any("text length" in name for name in tsv.feature_names))
        self.assertTrue(any("free text" in note for note in jsonl.notes))

    def test_large_files_are_sampled_repeatably(self):
        lines = ["first,second,target"]
        for index in range(90):
            lines.append(f"{index},{index % 7},{index % 2}")
        content = "\n".join(lines).encode()
        first = parse_tabular_upload(content, "large.csv", max_rows=35)
        second = parse_tabular_upload(content, "large.csv", max_rows=35)
        self.assertEqual(first.total_rows, 90)
        self.assertTrue(first.sampled)
        self.assertEqual(first.records, second.records)

    def test_xlsx_inspection_uses_first_visible_sheet(self):
        workbook = io.BytesIO()
        with zipfile.ZipFile(workbook, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                """<?xml version="1.0"?>
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets>
                </workbook>""",
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                """<?xml version="1.0"?>
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
                    Target="worksheets/sheet1.xml"/>
                </Relationships>""",
            )
            rows = [
                '<row r="1"><c r="A1" t="inlineStr"><is><t>first</t></is></c>'
                '<c r="B1" t="inlineStr"><is><t>second</t></is></c>'
                '<c r="C1" t="inlineStr"><is><t>target</t></is></c></row>'
            ]
            for index in range(1, 37):
                rows.append(
                    f'<row r="{index + 1}"><c r="A{index + 1}"><v>{index}</v></c>'
                    f'<c r="B{index + 1}"><v>{index % 5}</v></c>'
                    f'<c r="C{index + 1}"><v>{index % 2}</v></c></row>'
                )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                '<?xml version="1.0"?><worksheet '
                'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"<sheetData>{''.join(rows)}</sheetData></worksheet>",
            )
        inspected = inspect_tabular_upload(workbook.getvalue(), "signals.xlsx")
        self.assertEqual(inspected["format"], "Excel workbook")
        self.assertEqual(inspected["rows"], 36)
        self.assertEqual(inspected["columns"], ["first", "second", "target"])
        self.assertEqual(inspected["task"], "classification")


if __name__ == "__main__":
    unittest.main()
