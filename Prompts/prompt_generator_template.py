from langchain_core.prompts import PromptTemplate

#Copied: 1_dynamic_prompt

# Prompt template
template= PromptTemplate(
    template="""You are an expert writer.Write a {paper_input} in a {style_input} style.Ensure the content is well-structured, coherent, and aligns with the conventions of a typical .Respond only with the complete {length_input} content.
    """,
    input_variables=['paper_input','style_input','length_input'],
    validate_template=True
)

template.save('template.json')