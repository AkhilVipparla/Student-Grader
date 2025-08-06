# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_community.chat_models import ChatOllama

# # Grading prompt
# grading_prompt = PromptTemplate(
#     input_variables=["context", "question", "answer"],
#     template="""
# You are a subject expert grading a student's response.

# Reference:
# {context}

# Question:
# {question}

# Student Answer:
# {answer}

# Instructions:
# 1. Gently compare the student’s answer with the reference. Highlight correct points, small mistakes, and any missing ideas.
# 2. Only refer to facts present in the reference. If something isn’t mentioned, say “Not in context.”
# 3. Give a balanced score out of 10, considering overall understanding even if minor details are missing. Provide a short reason.
# 4. Do not focus on grammar or wording style.
# 5. Accept the answer even if it is the paraphrased facts from the reference.

# Respond as:
# - Evaluation
# - Score: X/10
# - Feedback
# """
# )

# # UI
# st.title("Student Answer Grader")

# # Load saved context
# try:
#     with open("retrieved_context.txt", "r") as f:
#         context = f.read()
# except FileNotFoundError:
#     context = ""

# if not context:
#     st.error("❌ No context found. Please run retriever_app.py first.")
# else:
#     question = st.text_input("Enter Question (Same as before)")
#     student_answer = st.text_area("Student Answer")

#     if question and student_answer:
#         with st.spinner("Grading..."):
#             llm = ChatOllama(model="llama3", temperature=0)
#             chain = LLMChain(llm=llm, prompt=grading_prompt)
#             feedback = chain.run(context=context, question=question, answer=student_answer)

#         st.success("✅ Evaluation Complete")
#         st.subheader("Feedback")
#         st.markdown(feedback)

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0)


EVALUATOR_PROMPT = PromptTemplate.from_template("""
You are an evaluator comparing a student's answer to the reference.

Reference:
{context}

Question:
{question}

Student Answer:
{answer}

Your Task:
- List points the student got right.
- List what is missing or incorrect.
- Identify anything not in the reference.

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
    st.error("❌ No retrieved context found. Please run the retriever first.")


if st.button("Run Agentic Grading") and question and student_answer and context:
    with st.spinner("Evaluating..."):
        evaluation = evaluator_chain.run(context=context, question=question, answer=student_answer)
        print("evaluation",evaluation)
        grade = grader_chain.run(evaluation=evaluation)
        feedback = feedback_chain.run(evaluation=evaluation)
    

    st.subheader("Evaluation")
    st.markdown(evaluation)

    st.subheader("Grade")
    st.markdown(grade)

    st.subheader("Feedback")
    st.markdown(feedback)


