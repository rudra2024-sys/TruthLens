import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "truthlens.db"


def main():
    print("=" * 60)
    print("TruthLens image_analysis schema migration")
    print("=" * 60)

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        # Confirm existing rows before changing anything.
        rows = cursor.execute(
            """
            SELECT
                analysis_id,
                result_id,
                efficientnet_score,
                fft_score,
                fake_probability,
                real_probability
            FROM image_analysis
            """
        ).fetchall()

        print(f"Existing image_analysis rows: {len(rows)}")

        # Create the corrected table.
        cursor.execute(
            """
            CREATE TABLE image_analysis_new (
                analysis_id VARCHAR NOT NULL PRIMARY KEY,
                result_id VARCHAR NOT NULL,
                efficientnet_score FLOAT NULL,
                fft_score FLOAT NULL,
                fake_probability FLOAT NULL,
                real_probability FLOAT NULL,
                FOREIGN KEY(result_id)
                    REFERENCES detection_results(result_id)
            )
            """
        )

        # Copy every existing record without changing its values.
        cursor.execute(
            """
            INSERT INTO image_analysis_new (
                analysis_id,
                result_id,
                efficientnet_score,
                fft_score,
                fake_probability,
                real_probability
            )
            SELECT
                analysis_id,
                result_id,
                efficientnet_score,
                fft_score,
                fake_probability,
                real_probability
            FROM image_analysis
            """
        )

        copied_rows = cursor.rowcount

        # Replace the old table.
        cursor.execute("DROP TABLE image_analysis")

        cursor.execute(
            "ALTER TABLE image_analysis_new "
            "RENAME TO image_analysis"
        )

        connection.commit()

        # Verify.
        schema = cursor.execute(
            "PRAGMA table_info(image_analysis)"
        ).fetchall()

        final_rows = cursor.execute(
            """
            SELECT
                analysis_id,
                result_id,
                efficientnet_score,
                fft_score,
                fake_probability,
                real_probability
            FROM image_analysis
            """
        ).fetchall()

        print()
        print("Migration complete.")
        print("Rows copied:", copied_rows)
        print()
        print("New schema:")

        for column in schema:
            print(column)

        print()
        print("Data after migration:")

        for row in final_rows:
            print(row)

        print()
        print("Verification:")
        print("Original rows:", len(rows))
        print("Final rows:", len(final_rows))
        print("Data preserved:", rows == final_rows)

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()