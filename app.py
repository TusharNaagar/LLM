from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import sqlite3

import google.generativeai as genai


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name Inventory and has the following columns - NAME, "Invoice ID", "Purchase Item", Price, "Customer ID"
    \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM Inventory ;
    \nExample 2 - Tell me all the customer has buy phone?, 
    the SQL command will be something like this SELECT * FROM Inventory WHERE Purchase Item = 'Phone'; 
    also the sql code should not have ``` in beginning or end and sql word in output
    \nExample 3 - Provide me Invoice id of John?, 
    the SQL command will be something like this SELECT Invoice ID FROM Inventory WHERE NAME = 'John';

    """

]


st.set_page_config(page_title="I can Retrieve Any SQL query")
st.header("Gemini App To Retrieve SQL Data")

question=st.text_input("Input: ",key="input")

submit=st.button("Ask the question")

if submit:
    response=get_gemini_response(question,prompt)
    print(response)
    response=read_sql_query(response,"Inventory.db")
    st.subheader("The REsponse is")
    for row in response:
        print(row)
        st.header(row)