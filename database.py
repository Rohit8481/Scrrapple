import asyncio
import mysql.connector
from datetime import datetime
import os

import AllinOne


async def run_database():

    # ============================================================
    # 1. RUN ALLINONE
    # ============================================================

    print("\n==============================")
    print("RUNNING ALLINONE")
    print("==============================\n")

    await AllinOne.main()


    # ============================================================
    # 2. GET GENERATED DATA
    # ============================================================

    data = AllinOne.data
    responses = AllinOne.responses
    severity = AllinOne.severity
    links = AllinOne.links
    summary = AllinOne.summary


    print("\n==============================")
    print("ALLINONE RESULTS")
    print("==============================")

    print("Headlines:", len(data))
    print("Short:", len(responses))
    print("Severity:", len(severity))
    print("Links:", len(links))
    print("Summary:", len(summary))


    # ============================================================
    # 3. STOP IF NO NEWS
    # ============================================================

    if not data:
        print("\nNo news found. Database insertion skipped.")
        return


    # ============================================================
    # 4. DATABASE CONNECTION
    # ============================================================

    db = mysql.connector.connect(
        host=os.getenv("TIDB_HOST"),
        user=os.getenv("TIDB_USER"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
    )

    cur = db.cursor()


    try:

        # ========================================================
        # 5. DELETE NEWS OLDER THAN 24 HOURS
        # ========================================================

        cur.execute("""
            DELETE FROM news
            WHERE time < NOW() - INTERVAL 24 HOUR
        """)


        # ========================================================
        # 6. GET EXISTING HEADLINES
        # ========================================================

        cur.execute("""
            SELECT headline
            FROM news
        """)

        old_data = {
            row[0]
            for row in cur.fetchall()
        }


        # ========================================================
        # 7. GET NEXT ID
        # ========================================================

        cur.execute("""
            SELECT COALESCE(MAX(id), 100)
            FROM news
        """)

        next_id = cur.fetchone()[0] + 1


        # ========================================================
        # 8. CURRENT TIME
        # ========================================================

        current_time = datetime.now()


        # ========================================================
        # 9. INSERT QUERY
        # ========================================================

        query = """
            INSERT INTO news
            (
                id,
                headline,
                link,
                severity,
                short,
                source,
                summary,
                time
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """


        # ========================================================
        # 10. INSERT ONLY NEW HEADLINES
        # ========================================================

        inserted = 0
        skipped = 0


        for index, headline in enumerate(data):

            # ----------------------------------------------------
            # DUPLICATE CHECK
            # ----------------------------------------------------

            if headline in old_data:

                print("\nDuplicate skipped:")
                print(headline)

                skipped += 1

                continue


            # ----------------------------------------------------
            # SOURCE
            # ----------------------------------------------------

            link = links[index]

            if "ndtv.com" in link:
                source = "NDTV"

            elif "thehindu.com" in link:
                source = "The Hindu"

            elif "timesofindia" in link:
                source = "Times of India"

            elif "hindustantimes.com" in link:
                source = "Hindustan Times"

            elif "indianexpress.com" in link:
                source = "Indian Express"

            else:
                source = "Unknown"


            # ----------------------------------------------------
            # INSERT
            # ----------------------------------------------------

            cur.execute(
                query,
                (
                    next_id,
                    headline,
                    link,
                    severity[index],
                    responses[index],
                    source,
                    summary[index],
                    current_time,
                )
            )


            print("\nInserted:")
            print("ID:", next_id)
            print("Headline:", headline)
            print("Severity:", severity[index])
            print("Source:", source)


            # Add to set so duplicates within this same run
            # are also prevented.
            old_data.add(headline)

            next_id += 1
            inserted += 1


        # ========================================================
        # 11. SAVE CHANGES
        # ========================================================

        db.commit()


        # ========================================================
        # 12. FINAL RESULT
        # ========================================================

        print("\n==============================")
        print("DATABASE COMPLETE")
        print("==============================")

        print("Inserted:", inserted)
        print("Skipped:", skipped)

        print("==============================\n")


    except Exception as e:

        db.rollback()

        print("\nDATABASE ERROR:")
        print(e)

        raise


    finally:

        cur.close()
        db.close()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        run_database()
    )
