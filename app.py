import streamlit as st
import mysql.connector
import datetime
import pandas as pd

############################
# MySQL Configuration
############################
def get_db_connection():
    """
    Connect to your Hostinger MySQL database using the credentials you've provided.
    """
    # Directly using your credentials here.
    # If you're making the repo private, that’s acceptable short-term.
    # Long-term, consider using Streamlit secrets to avoid exposing the password.
    host = "127.0.0.1"           # or "localhost" as displayed in phpMyAdmin
    port = 3306                  # MySQL default port
    user = "u628260032_francisdavid"
    password = "Chennai@202475"  # Provided password
    database = "u628260032_academapp"

    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        return conn
    except Exception as e:
        st.error(f"Could not connect to MySQL: {e}")
        return None

def log_usage_to_db(log_entry):
    """
    Insert a log entry into usage_logs table.
    log_entry might look like:
    {
      "screen_index": 3,
      "changes": {
         "Screen Title": {"old":"Old Title","new":"New Title"},
         "Text": {"old":"...","new":"..."}
      }
    }
    """
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        # Convert the changes dict to string (could also use json.dumps)
        changes_str = str(log_entry.get("changes", {}))
        log_time = datetime.datetime.now()

        sql = """
        INSERT INTO usage_logs (log_time, screen_index, changes)
        VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (log_time, log_entry.get("screen_index"), changes_str))
        conn.commit()

        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error inserting usage log: {e}")

############################
# Example Lesson Builder Code
############################

def main():
    st.set_page_config(page_title="Lesson Builder Demo", layout="wide")

    if "screens_df" not in st.session_state:
        # A simple DataFrame representing auto-generated screens
        st.session_state["screens_df"] = pd.DataFrame(
            [
                {"Screen Number":1, "Screen Title":"Intro", "Text":"Welcome text", "Estimated Duration":"2 minutes"},
                {"Screen Number":2, "Screen Title":"Key Concept", "Text":"Concept text", "Estimated Duration":"2 minutes"}
            ]
        )

    st.title("Demo: MySQL Logging on Hostinger")

    df = st.session_state["screens_df"]
    st.write("Current screens:")
    st.dataframe(df)

    if st.button("Refine & Log Changes"):
        # For demonstration, let's pretend we changed the second screen's text
        old_row = df.loc[1].copy()
        df.at[1, "Text"] = "Updated Concept text"
        # Now log the difference
        log_user_change_db(1, old_row, df.loc[1])

        st.success("Changes applied & usage logged to DB!")
        st.dataframe(df)

def log_user_change_db(screen_index, old_row, new_row):
    """
    Compare old vs. new row, log only if there's a difference.
    """
    changes = {}
    for col in ["Screen Title", "Text", "Estimated Duration"]:
        old_val = old_row[col]
        new_val = new_row[col]
        if old_val != new_val:
            changes[col] = {"old": old_val, "new": new_val}

    if changes:
        log_entry = {
            "screen_index": screen_index,
            "changes": changes
        }
        log_usage_to_db(log_entry)

if __name__ == "__main__":
    main()
