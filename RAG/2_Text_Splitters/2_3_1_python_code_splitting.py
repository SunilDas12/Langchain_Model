from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

code="""
class Employee:
    def __init__(self, emp_id, name, department, salary):
        self.emp_id = emp_id
        self.name = name
        self.department = department
        self.salary = salary

    def display_info(self):
        print(f"ID: {self.emp_id}, Name: {self.name}, "
              f"Department: {self.department}, Salary: {self.salary}")

    def update_salary(self, new_salary):
        self.salary = new_salary
        print(f"Salary updated for {self.name}: {self.salary}")

emp1 = Employee(101, "Amit Sharma", "IT", 65000)
emp2 = Employee(102, "Priya Verma", "HR", 72000)

emp1.display_info()
emp2.display_info()

emp1.update_salary(70000)
emp1.display_info()

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # You can change the language based on requirements
    chunk_size=500,
    chunk_overlap=0,
)

chunks=splitter.split_text(code)

print(len(chunks))
print(chunks[0])
print('------------------')
print(chunks[1])