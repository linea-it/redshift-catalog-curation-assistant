import pytest
from astropy.io import fits
from astropy.table import Table

import redshift_catalog_curation_assistant.fits.fits as fits_module
from redshift_catalog_curation_assistant.fits import (
    LargeCompressedFitsError,
    describe_fits,
    format_fits_description,
)


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


def test_describe_fits_does_not_access_hdu_data(tmp_path, monkeypatch):
    """Ensure FITS description is header-only for large memmapped files."""
    path = tmp_path / "sample.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(Table({"object_id": [1, 2], "z": [0.1, 0.2]}), name="CATALOG"),
        ]
    ).writeto(path)

    original_open = fits.open

    class HeaderOnlyHDU:
        def __init__(self, hdu):
            self._hdu = hdu
            self.header = hdu.header
            self.name = hdu.name

        @property
        def data(self):
            raise AssertionError("describe_fits should not access hdu.data")

    class HeaderOnlyHDUList:
        def __init__(self, hdul):
            self._hdul = hdul

        def __enter__(self):
            return [HeaderOnlyHDU(hdu) for hdu in self._hdul]

        def __exit__(self, exc_type, exc, traceback):
            self._hdul.close()

    def header_only_open(*args, **kwargs):
        return HeaderOnlyHDUList(original_open(*args, **kwargs))

    monkeypatch.setattr(fits, "open", header_only_open)

    summaries = describe_fits(path, max_columns=1)

    assert summaries[1]["n_rows"] == 2
    assert summaries[1]["n_columns"] == 2
    assert summaries[1]["columns"] == ["object_id"]


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


def test_describe_fits_rejects_large_compressed_fits_before_opening(tmp_path, monkeypatch):
    """Ensure large .fits.gz files fail before Astropy tries to open them."""
    path = tmp_path / "large.fits.gz"
    path.write_bytes(b"not a real gzip")
    monkeypatch.setattr(fits_module, "_gzip_uncompressed_size", lambda path: 3 * 1024 * 1024 * 1024)

    with pytest.raises(LargeCompressedFitsError, match="gzip -dk"):
        describe_fits(path)


def test_describe_fits_rejects_large_compressed_fits_by_compressed_size(tmp_path, monkeypatch):
    """Ensure large compressed files are rejected even when gzip size wraps."""
    path = tmp_path / "large.fits.gz"
    path.write_bytes(b"not a real gzip")
    monkeypatch.setattr(fits_module, "_gzip_uncompressed_size", lambda path: 1)
    monkeypatch.setattr(
        fits_module,
        "LARGE_COMPRESSED_FITS_COMPRESSED_THRESHOLD_BYTES",
        1,
    )

    with pytest.raises(LargeCompressedFitsError, match="Compressed size"):
        describe_fits(path)
