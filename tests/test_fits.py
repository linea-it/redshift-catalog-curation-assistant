from astropy.io import fits
from astropy.table import Table

from redshift_catalog_curation_assistant.fits import describe_fits, format_fits_description


def test_describe_fits_lists_hdus_and_table_columns(tmp_path):
    """Verify FITS HDU descriptions include table shape and column previews."""
    path = tmp_path / "sample.fits"
    table = Table({"object_id": [1, 2], "z": [0.1, 0.2], "quality": [3, 4]})
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=[[1, 2], [3, 4]], name="IMAGE"),
            fits.BinTableHDU(table, name="CATALOG"),
        ]
    )
    hdul.writeto(path)

    summaries = describe_fits(path, max_columns=2)

    assert [summary["index"] for summary in summaries] == [0, 1, 2]
    assert summaries[2]["name"] == "CATALOG"
    assert summaries[2]["n_rows"] == 2
    assert summaries[2]["n_columns"] == 3
    assert summaries[2]["columns"] == ["object_id", "z"]
    assert summaries[2]["truncated_columns"] == 1


def test_format_fits_description_includes_hdu_and_column_preview(tmp_path):
    """Verify CLI formatting includes the details needed to choose fits_hdu."""
    summaries = [
        {"index": 0, "name": "PRIMARY", "type": "PrimaryHDU", "shape": None},
        {
            "index": 1,
            "name": "CATALOG",
            "type": "BinTableHDU",
            "shape": (2,),
            "n_rows": 2,
            "n_columns": 3,
            "columns": ["object_id", "z"],
            "truncated_columns": 1,
        },
    ]

    output = format_fits_description(summaries)

    assert "HDU 1: CATALOG" in output
    assert "rows=2 columns=3" in output
    assert "object_id, z, ... (+1 more)" in output
