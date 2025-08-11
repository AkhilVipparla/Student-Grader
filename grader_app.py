
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0.6, stop=["<|eot_id|>"])


EVALUATOR_PROMPT = PromptTemplate.from_template("""
You are an evaluator evaluating students answer.

Reference:
{context}

Question:
{question}


{answer}


Your Task:
- List points the student got right.
- List what is missing or incorrect if mentioned in the context.
- Carefully evaluate each point in the answer, including the final one. Do not skip any.


Be objective and clear.
""")
evaluator_chain = LLMChain(llm=llm, prompt=EVALUATOR_PROMPT)
#print("evaluator chain",evaluator_chain)

GRADER_PROMPT = PromptTemplate.from_template("""
You are a grader assigning a score based on the evaluation below.

Evaluation:
{evaluation}
                                             
                                             

Tasks:
- Give a score out of 10.
- Justify the score based on the evaluation.
- Be fair and do not deduct for grammar.

Format:
Score: X/10
Reason: ...
""")
grader_chain = LLMChain(llm=llm, prompt=GRADER_PROMPT)


FEEDBACK_PROMPT = PromptTemplate.from_template("""
Based on the evaluation below, suggest two things the student should improve.

Evaluation:
{evaluation}

Respond with practical tips.
""")
feedback_chain = LLMChain(llm=llm, prompt=FEEDBACK_PROMPT)

# Streamlit UI
st.title("Student Grader")

question = st.text_input("Enter Question")
student_answer = st.text_area("Enter Student Answer")
#print("student_answer",student_answer)
try:
    with open("retrieved_context.txt", "r") as f:
        context = f.read()
    
except FileNotFoundError:
    context = ""
    st.error("No retrieved context found. Please run the retriever first.")


if question and student_answer and context:
    with st.spinner("Evaluating..."):
        evaluation = evaluator_chain.run(context=context, question=question, answer=student_answer.strip())
        #("evaluation",evaluation)
        grade = grader_chain.run(evaluation=evaluation)
        feedback = feedback_chain.run(evaluation=evaluation)
    

    st.subheader("Evaluation")
    st.markdown(evaluation)

    st.subheader("Grade")
    st.markdown(grade)

    st.subheader("Feedback")
    st.markdown(feedback)
