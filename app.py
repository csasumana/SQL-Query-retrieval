import os
import re
import mysql.connector
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Model and provide queries as response
def get_gemini_response(question, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt, question])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response from Gemini model: {e}")
        return None

# Function to retrieve query from the database
def read_sql_query(sql, db):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database=db
        )
        cur = conn.cursor(dictionary=True)  # Use dictionary cursor for easier result handling
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except mysql.connector.Error as e:
        st.error(f"MySQL error: {e.msg}")
        return None

# Function to retrieve query from the database
def execute_query(query, host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute(f"USE {database};")  # Explicitly select the database
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        print(f"Database error: {e}")
        if e.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        elif e.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print("Username or password is incorrect")
        else:
            print(f"Error: {e}")
        return None

# Define Few-Shot Examples
few_shot_examples = [
    {
        "question": "What is the average rental duration for all films?",
        "sql_query": "SELECT AVG(rental_duration) AS average_rental_duration FROM film;",
        "expected_result": "The average rental duration for all films is 4.98 days."
    },
    {
        "question": "List all the film titles.",
        "sql_query": "SELECT title FROM film;",
        "expected_result": "Here are the titles of all films in the database."
    },
    {
        "question": "How many films are there in each category?",
        "sql_query": """
            SELECT category.name AS category_name, COUNT(f.film_id) AS num_films
            FROM category c
            JOIN film_category fc ON c.category_id = fc.category_id
            JOIN film f ON fc.film_id = f.film_id
            GROUP BY c.name;
        """,
        "expected_result": "Here is the count of films in each category."
    },
    {
        "question": "Show the top 5 films with the longest rental duration.",
        "sql_query": """
            SELECT title, rental_duration
            FROM film
            ORDER BY rental_duration DESC
            LIMIT 5;
        """,
        "expected_result": "Here are the top 5 films with the longest rental duration."
    },
    {
        "question": "What is the total revenue generated from film rentals?",
        "sql_query": """
            SELECT SUM(p.amount) AS total_revenue
            FROM payment p
            JOIN rental r ON p.rental_id = r.rental_id;
        """,
        "expected_result": "The total revenue generated from film rentals is $XXXX."
    }
]

# Construct Prompt Template for Few-Shot Learning
def construct_prompt_template(few_shot_examples):
    prompt = """
        You are interacting with the Sakila database, which contains several key tables:
        
        1. **Actor:**
           - `actor_id` (INT): Primary key, unique identifier for each actor.
           - `first_name` (VARCHAR): First name of the actor.
           - `last_name` (VARCHAR): Last name of the actor.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        2. **Film:**
           - `film_id` (INT): Primary key, unique identifier for each film.
           - `title` (VARCHAR): Title of the film.
           - `description` (TEXT): Description or summary of the film.
           - `release_year` (YEAR): Year of the film's release.
           - `language_id` (TINYINT): Foreign key referencing `language.language_id`.
           - `rental_duration` (TINYINT): Number of days the film can be rented.
           - `rental_rate` (DECIMAL): Rental rate per day for the film.
           - `length` (SMALLINT): Duration of the film in minutes.
           - `replacement_cost` (DECIMAL): Cost to replace the film if lost or damaged.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        3. **Film_Actor:**
           - `actor_id` (INT): Foreign key referencing `actor.actor_id`.
           - `film_id` (INT): Foreign key referencing `film.film_id`.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        4. **Category:**
           - `category_id` (INT): Primary key, unique identifier for each category.
           - `name` (VARCHAR): Name of the category.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        5. **Film_Category:**
           - `film_id` (INT): Foreign key referencing `film.film_id`.
           - `category_id` (INT): Foreign key referencing `category.category_id`.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        6. **Language:**
           - `language_id` (TINYINT): Primary key, unique identifier for each language.
           - `name` (CHAR): Name of the language.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        7. **Inventory:**
           - `inventory_id` (INT): Primary key, unique identifier for each inventory item.
           - `film_id` (INT): Foreign key referencing `film.film_id`.
           - `store_id` (INT): Foreign key referencing `store.store_id`.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        8. **Store:**
           - `store_id` (INT): Primary key, unique identifier for each store.
           - `manager_staff_id` (INT): Foreign key referencing `staff.staff_id`.
           - `address_id` (INT): Foreign key referencing `address.address_id`.
           - `last_update` (TIMESTAMP): Timestamp of the last update.

        9. **Staff:**
            - `staff_id` (INT): Primary key, unique identifier for each staff member.
            - `first_name` (VARCHAR): First name of the staff member.
            - `last_name` (VARCHAR): Last name of the staff member.
            - `address_id` (INT): Foreign key referencing `address.address_id`.
            - `email` (VARCHAR): Email address of the staff member.
            - `store_id` (INT): Foreign key referencing `store.store_id`.
            - `active` (BOOLEAN): Indicates if the staff member is active.
            - `username` (VARCHAR): Username for staff login.
            - `password` (VARCHAR): Password for staff login.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        10. **Address:**
            - `address_id` (INT): Primary key, unique identifier for each address.
            - `address` (VARCHAR): Street address.
            - `district` (VARCHAR): District or region.
            - `city_id` (INT): Foreign key referencing `city.city_id`.
            - `postal_code` (VARCHAR): Postal code.
            - `phone` (VARCHAR): Phone number.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        11. **City:**
            - `city_id` (INT): Primary key, unique identifier for each city.
            - `city` (VARCHAR): Name of the city.
            - `country_id` (INT): Foreign key referencing `country.country_id`.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        12. **Country:**
            - `country_id` (INT): Primary key, unique identifier for each country.
            - `country` (VARCHAR): Name of the country.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        13. **Customer:**
            - `customer_id` (INT): Primary key, unique identifier for each customer.
            - `store_id` (INT): Foreign key referencing `store.store_id`.
            - `first_name` (VARCHAR): First name of the customer.
            - `last_name` (VARCHAR): Last name of the customer.
            - `email` (VARCHAR): Email address of the customer.
            - `address_id` (INT): Foreign key referencing `address.address_id`.
            - `active` (BOOLEAN): Indicates if the customer is active.
            - `create_date` (DATETIME): Date the customer was created.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        14. **Rental:**
            - `rental_id` (INT): Primary key, unique identifier for each rental.
            - `rental_date` (DATETIME): Date and time the rental was made.
            - `inventory_id` (INT): Foreign key referencing `inventory.inventory_id`.
            - `customer_id` (INT): Foreign key referencing `customer.customer_id`.
            - `return_date` (DATETIME): Date and time the rental was returned.
            - `staff_id` (INT): Foreign key referencing `staff.staff_id`.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        15. **Payment:**
            - `payment_id` (INT): Primary key, unique identifier for each payment.
            - `customer_id` (INT): Foreign key referencing `customer.customer_id`.
            - `staff_id` (INT): Foreign key referencing `staff.staff_id`.
            - `rental_id` (INT): Foreign key referencing `rental.rental_id`.
            - `amount` (DECIMAL): Amount of the payment.
            - `payment_date` (DATETIME): Date and time of the payment.
            - `last_update` (TIMESTAMP): Timestamp of the last update.

        You will be provided with questions, and your task is to generate the appropriate SQL query to retrieve the required data from the Sakila database.

        Here are some examples:

    """
    for example in few_shot_examples:
        prompt += f"    \n\nQuestion: {example['question']}\n    SQL Query: {example['sql_query']}\n    Expected Result: {example['expected_result']}"
    
    return prompt

# Streamlit App Interface
st.title("Sakila Database Query Assistant")
question = st.text_input("Enter your question about the Sakila database:")
prompt_template = construct_prompt_template(few_shot_examples)

if st.button("Generate Query"):
    if question:
        # Get Gemini response
        gemini_response = get_gemini_response(question, prompt_template)
        if gemini_response:
            st.markdown("### Generated SQL Query:")
            st.code(gemini_response, language="sql")
            # Extract SQL query using regex (improve regex as needed)
            match = re.search(r"(WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+.*?;", gemini_response, re.DOTALL)
            if match:
                sql_query = match.group(0)
                st.markdown("### Extracted SQL Query:")
                st.code(sql_query, language="sql")
                # Execute SQL query on Sakila database
                results = execute_query(sql_query, "localhost", "root", "root", "sakila1")
                if results:
                    st.markdown("### Query Results:")
                    st.write(results)
                else:
                    st.error("No results found or error executing query.")
            else:
                st.error("Could not extract SQL query from Gemini response.")
        else:
            st.error("No response generated from Gemini model.")
    else:
        st.warning("Please enter a question.")
