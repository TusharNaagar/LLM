import sqlite3

connection=sqlite3.connect("Inventory.db")

cursor=connection.cursor()

table_info="""
Create table Inventory (NAME VARCHAR(25),"Invoice ID" VARCHAR(25),"Purchase Item" VARCHAR(25), Price INT, "Customer ID" VARCHAR(25));

"""
cursor.execute(table_info)

cursor.execute('''Insert Into Inventory values('John', 'INV001', 'Laptop', 1500, 'CUST001')''')
cursor.execute('''Insert Into Inventory values('Alice', 'INV002', 'Phone', 800, 'CUST002')''')
cursor.execute('''Insert Into Inventory values('Bob', 'INV003', 'Headphones', 100, 'CUST003')''')
cursor.execute('''Insert Into Inventory values('Eva', 'INV004', 'Tablet', 600, 'CUST004')''')
cursor.execute('''Insert Into Inventory values('David', 'INV005', 'Keyboard', 50, 'CUST005')''')


print("The inserted records are")
data=cursor.execute('''Select * from Inventory''')
for row in data:
    print(row)

connection.commit()
connection.close()