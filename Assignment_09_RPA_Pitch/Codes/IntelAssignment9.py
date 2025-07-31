#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import datetime
import pdfplumber
import os

invoice_log = []

def extract_invoice_data(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text()

            invoice_number = "N/A"
            vendor_name = "N/A"
            amount = "N/A"
            for line in text.split('\n'):
                if "Invoice No" in line:
                    invoice_number = line.split(":")[-1].strip()
                elif "Vendor" in line:
                    vendor_name = line.split(":")[-1].strip()
                elif "Total" in line:
                    amount = line.split(":")[-1].strip()

            return {
                "Invoice Number": invoice_number,
                "Vendor Name": vendor_name,
                "Amount": amount,
                "Date": datetime.datetime.today().strftime('%d-%m-%Y')
            }
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read PDF:\n{e}")
        return None

def simulate_invoice_upload():
    file_path = filedialog.askopenfilename(title="Select Invoice PDF", filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        data = extract_invoice_data(file_path)
        if data:
            invoice_number_var.set(data["Invoice Number"])
            vendor_name_var.set(data["Vendor Name"])
            amount_var.set(data["Amount"])
            date_var.set(data["Date"])
            messagebox.showinfo("Invoice Uploaded", f"Extracted data from:\n{os.path.basename(file_path)}")

def submit_to_erp():
    data = {
        "Invoice Number": invoice_number_var.get(),
        "Vendor Name": vendor_name_var.get(),
        "Amount": amount_var.get(),
        "Date": date_var.get()
    }
    if all(data.values()):
        invoice_log.append(data)
        update_dashboard()
        messagebox.showinfo("Success", "Invoice successfully submitted to ERP backend.")
        clear_form()
    else:
        messagebox.showwarning("Missing Data", "Please fill in all fields before submitting.")

def clear_form():
    invoice_number_var.set("")
    vendor_name_var.set("")
    amount_var.set("")
    date_var.set("")

def update_dashboard():
    for row in dashboard_table.get_children():
        dashboard_table.delete(row)
    for idx, log in enumerate(invoice_log):
        dashboard_table.insert("", "end", iid=idx, values=(
            log["Invoice Number"], log["Vendor Name"], log["Amount"], log["Date"]
        ))

# GUI
root = tk.Tk()
root.title("Tata Steel RPA – Invoice Automation System")
root.geometry("700x600")
root.configure(bg="#e6f0ff")

title_label = tk.Label(root, text="Tata Steel – RPA Invoice Entry Automation", font=("Segoe UI", 16, "bold"),
                       fg="#003366", bg="#e6f0ff", pady=10)
title_label.pack()

upload_btn = tk.Button(root, text="Upload Invoice (PDF)", command=simulate_invoice_upload,
                       font=("Segoe UI", 12), bg="#004080", fg="white", width=25)
upload_btn.pack(pady=10)

invoice_number_var = tk.StringVar()
vendor_name_var = tk.StringVar()
amount_var = tk.StringVar()
date_var = tk.StringVar()

def create_field(label, variable):
    frame = tk.Frame(root, bg="#e6f0ff")
    tk.Label(frame, text=label, font=("Segoe UI", 12), width=15, anchor='w', bg="#e6f0ff").pack(side="left")
    tk.Entry(frame, textvariable=variable, font=("Segoe UI", 12), width=40).pack(side="left")
    frame.pack(pady=5)

create_field("Invoice Number:", invoice_number_var)
create_field("Vendor Name:", vendor_name_var)
create_field("Amount:", amount_var)
create_field("Date:", date_var)

submit_btn = tk.Button(root, text="Submit to ERP", command=submit_to_erp,
                       font=("Segoe UI", 12, "bold"), bg="#0066cc", fg="white", width=30)
submit_btn.pack(pady=15)

# Dashboard Table
tk.Label(root, text="ERP Invoice Log", font=("Segoe UI", 12, "bold"), bg="#e6f0ff", fg="#003366").pack(pady=5)
dashboard_frame = tk.Frame(root, bg="#e6f0ff")
dashboard_frame.pack()

columns = ("Invoice Number", "Vendor Name", "Amount", "Date")
dashboard_table = ttk.Treeview(dashboard_frame, columns=columns, show="headings", height=8)
for col in columns:
    dashboard_table.heading(col, text=col)
    dashboard_table.column(col, anchor="center", width=150)
dashboard_table.pack()

footer = tk.Label(root, text="© 2025 Tata Steel | RPA Simulation Dashboard", font=("Segoe UI", 9), bg="#e6f0ff", fg="#666666")
footer.pack(side="bottom", pady=10)

root.mainloop()


# In[ ]:




