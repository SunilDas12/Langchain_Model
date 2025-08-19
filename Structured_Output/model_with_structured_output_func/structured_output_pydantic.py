from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
import os

load_dotenv()

model=AzureChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        model_name=os.environ['OPENAI_AZURE_MODEL'],
        temperature=0.4,
        max_tokens=1000,
        seed=42,
)

class Review(BaseModel):
    key_themes: list[str] = Field(description="List of key themes extracted from the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos","neg"] = Field(description="Sentiment of the review, either positive or negative")
    pros:Optional[list[str]] = Field(default=None, description="List of all pros aspects mentioned in the review")
    cons: Optional[list[str]] = Field(default=None, description="List of all cons aspects mentioned in the review")
    name: Optional[str] = Field(default=None, description="Name of the reviewer, if available")

structured_model = model.with_structured_output(Review)

result= structured_model.invoke(
    """I have been using your product for the past 4 months, and while I genuinely appreciate the innovation and ambition behind it, my experience has been a bit of a rollercoaster. Let's start with the positives — the UI is sleek, intuitive, and clearly designed with the user in mind. It made onboarding fairly smooth, and I loved the initial setup walkthrough. Also, your customer support team deserves kudos — responses were prompt and thoughtful, especially when I encountered some billing discrepancies.

    However, the functionality itself has fallen short in several areas. The analytics dashboard, which was one of the main reasons I signed up, is often laggy and refreshes inconsistently, leading to incomplete or outdated data being shown. This was particularly frustrating during our monthly reporting cycles. In addition, the mobile experience — while visually consistent with desktop — feels like an afterthought. Certain features are missing, and responsiveness is lacking.

    Another major pain point was the integration with our CRM. Despite being marketed as 'seamless', we encountered frequent sync failures and missing data entries. We had to develop custom scripts as workarounds, which added unexpected overhead. To be honest, this made us question the value proposition.

    I still believe in the products potential, and I can see the roadmap going in the right direction, especially with the recent beta features. But unless core stability issues are addressed, it will be hard for us to justify renewal when our subscription ends in two months. Please take this as constructive feedback — I am rooting for your team to succeed""")

print(f"Key_themes: {result.key_themes}")
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")
print(f"Pros: {result.pros}")
print(f"Cons: {result.cons}")
print(f"Name: {result.name}")