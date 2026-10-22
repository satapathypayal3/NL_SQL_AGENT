"""Create a sample retail database (SQLite) for the agent to query.

Run:  python data/create_db.py
Creates data/retail.db with four tables: customers, products, orders, order_items.
The data is random but reproducible (fixed seed), so evaluation results are stable.
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "retail.db"

SCHEMA = """
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    city          TEXT NOT NULL,
    signup_date   DATE NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    price         REAL NOT NULL          -- price in INR
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    DATE NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('delivered', 'shipped', 'cancelled', 'returned'))
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL          -- price paid per unit in INR
);
"""

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Kabir", "Arjun", "Sai",
               "Ananya", "Diya", "Isha", "Kavya", "Meera", "Priya", "Riya", "Sneha",
               "Neha", "Pooja", "Rahul", "Vikram", "Tanvi", "Nikhil", "Shreya", "Aditi"]
LAST_NAMES = ["Sharma", "Patil", "Kulkarni", "Deshmukh", "Iyer", "Nair", "Reddy", "Gupta",
              "Joshi", "Mehta", "Shah", "Das", "Mishra", "Singh", "Rao", "Kapoor"]
CITIES = ["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad"]

PRODUCTS = {
    "Electronics": [("Wireless Earbuds", 1999), ("Bluetooth Speaker", 2499), ("Smartwatch", 4999),
                    ("Power Bank 20000mAh", 1499), ("USB-C Charger", 899), ("Laptop Stand", 1299)],
    "Books": [("Python for Data Analysis", 899), ("Statistics Made Simple", 549),
              ("SQL Cookbook", 749), ("Deep Learning Basics", 999), ("The Data Story", 449)],
    "Home & Kitchen": [("Steel Water Bottle", 599), ("Non-stick Pan", 1199), ("Pressure Cooker 5L", 2299),
                       ("Mixer Grinder", 3499), ("Dinner Set", 1799)],
    "Clothing": [("Cotton Kurta", 999), ("Denim Jacket", 2199), ("Running Shoes", 2999),
                 ("Track Pants", 799), ("Formal Shirt", 1299)],
    "Sports": [("Yoga Mat", 699), ("Cricket Bat", 2599), ("Badminton Racket", 1499),
               ("Football", 899), ("Dumbbell Set 10kg", 1999)],
}

STATUS_WEIGHTS = {"delivered": 0.78, "shipped": 0.08, "cancelled": 0.09, "returned": 0.05}


def build(db_path: Path = DB_PATH, seed: int = 42) -> Path:
    rng = random.Random(seed)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # Customers
    start = date(2024, 1, 1)
    customers = []
    for cid in range(1, 201):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        city = rng.choices(CITIES, weights=[22, 20, 16, 14, 10, 8, 5, 5])[0]
        signup = start + timedelta(days=rng.randint(0, 540))
        customers.append((cid, name, city, signup.isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

    # Products
    products = []
    pid = 1
    for category, items in PRODUCTS.items():
        for pname, price in items:
            products.append((pid, pname, category, float(price)))
            pid += 1
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    # Orders and order items
    orders, items = [], []
    item_id = 1
    statuses, weights = zip(*STATUS_WEIGHTS.items())
    # About 7% of customers sign up but never order.
    active = rng.sample(customers, k=186)
    for oid in range(1, 1501):
        cust = rng.choice(active)
        signup = date.fromisoformat(cust[3])
        last_day = date(2026, 6, 30)
        span = max((last_day - signup).days, 1)
        odate = signup + timedelta(days=rng.randint(0, span))
        status = rng.choices(statuses, weights=weights)[0]
        orders.append((oid, cust[0], odate.isoformat(), status))

        for prod in rng.sample(products, k=rng.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]):
            qty = rng.choices([1, 2, 3], weights=[75, 20, 5])[0]
            discount = rng.choice([1.0, 1.0, 1.0, 0.9, 0.85])  # occasional sale price
            items.append((item_id, oid, prod[0], qty, round(prod[3] * discount, 2)))
            item_id += 1

    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)
    conn.commit()
    conn.close()
    return db_path


if __name__ == "__main__":
    path = build()
    print(f"Created {path}")
