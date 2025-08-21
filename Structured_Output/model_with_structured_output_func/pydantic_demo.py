from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# pydantic object
class Student(BaseModel):
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

new_student={'name':'sunil', 'email':'abc@gmail.com', 'cgpa':5}
#new_student={'name':10} #### Error: Input should be a valid string [type=string_type, input_value=10, input_type=int]

student=Student(**new_student)
print(student)

student_dict=dict(student)
print(student_dict['age'])

student_json=student.model_dump_json()
print(student_json)