from galileo import GalileoLogger, galileo_context
import os
import time
from datetime import datetime
import logging
from dotenv import load_dotenv
from galileo import galileo_context
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment

load_dotenv()

logger_debug = logging.getLogger(__name__)

def main():
    dataset = get_dataset(name="trades")
    with galileo_context():
        results = run_experiment(
            "finance-chat-test2",
            dataset=dataset,
            function=log_hallucination,
            metrics=["correctness"],
        project="finance-chat-testing",
    )
    


def log_hallucination(input=None):
    """
    Log a sample hallucination to Galileo for demonstration and testing purposes.
    
    Args:
        project_name: The Galileo project name to log to
        log_stream: The Galileo log stream to log to
    """
    logger = galileo_context.get_logger_instance()

    # Start a workflow trace
    print("Starting workflow trace...")

    # Add retriever span
    print("Adding retriever span...")
    logger.add_retriever_span(
        input="What was Broadcom\'s revenue in Q4 and how did it compare to the previous quarter?",
        output=[
"""Company Name: Broadcom | Quarter: Q4 | 

Kirsten Spears
Chief Financial Officer and Chief Accounting Officer at Broadcom
Thank you, Hock. Let me now provide additional detail on our Q4 financial performance. Consolidated revenue was $9.3 billion for the quarter, up 4% from a year ago. Gross margins were 74.3% of revenue in the quarter, in line with our expectations. Operating expenses were $1.2 billion, flat year-on-year. R&D of $940 million was also stable year-on-year.
Operating income for the quarter was $5.7 billion and was up 4% from a year ago, with operating margin at 62% of revenue. Adjusted EBITDA was $6 billion or 65% of revenue, in line with expectations. This figure excludes $124 million of depreciation. Now a review of the P&L for our two segments, starting with our semiconductor segment.
Revenue for our Semiconductor Solutions segment was""",
"""Company Name: Broadcom | Quarter: Q4 | % year-on-year with Infrastructure Software operating margin at 75%. Now moving on to cash flow.
Free cash flow in the quarter was $4.7 billion and represented 51% of revenues in Q4. We spent $105 million on capital expenditures. Days sales outstanding were 31 days in the fourth quarter compared to 30 days in the third. We ended the fourth quarter with inventory of $1.9 billion, up 3% sequentially. We continue to remain disciplined on how we manage inventory across the ecosystem. We exited the quarter with 76 days of inventory on hand, down 80 days in Q3. We ended the fourth quarter with $14.2 billion of cash and $39.2 billion of gross debt, of which $1.6 billion is short term. Now let me recap our financial performance for fiscal 2023.
Our revenue had a record $35.8 billion, growing 8% year-on-year. Semiconductor revenue was $28.2""",
"""Company Name: Broadcom | Quarter: Q4 |  segments, starting with our semiconductor segment.
Revenue for our Semiconductor Solutions segment was $7.3 billion and represented 79% of total revenue in the quarter. This was up 3% year-on-year. Gross margins for our Semiconductor Solutions segment were approximately 70%, down 110 basis points year-on-year driven primarily by product mix within our semiconductor end markets. Operating expenses were stable year-on-year at $822 million, resulting in operating profit growth of 2% year-on-year and semiconductor operating margins of 58%.
Now moving on to our Infrastructure Software segment. Revenue for Infrastructure Software was $2 billion, up 7% year-on-year and represented 21% of revenue. Gross margins for Infrastructure Software were 92% in the quarter, and operating expenses were $339 million in the quarter. Q4 operating profit grew 12% year-on-year with Infrastructure Software operating margin at 75%. Now moving on to cash flow"""],
        name="RAG Retrieval",
        duration_ns=int(1.3e8),
        status_code=200
    )

    # Add LLM span for generating travel recommendations
    print("Adding LLM span...")
    logger.add_llm_span(
        input="""Human: You are a helpful assistant. Given the context below, please answer the following questions:

Company Name: Broadcom | Quarter: Q4 | 
Kirsten Spears Chief Financial Officer and Chief Accounting Officer at Broadcom Thank you, Hock. Let me now provide additional detail on our Q4 financial performance. Consolidated revenue was $9.3 billion for the quarter, up 4% from a year ago. Gross margins were 74.3% of revenue in the quarter, in line with our expectations. Operating expenses were $1.2 billion, flat year-on-year. R&D of $940 million was also stable year-on-year. Operating income for the quarter was $5.7 billion and was up 4% from a year ago, with operating margin at 62% of revenue. Adjusted EBITDA was $6 billion or 65% of revenue, in line with expectations. This figure excludes $124 million of depreciation. Now a review of the P&L for our two segments, starting with our semiconductor segment. Revenue for our Semiconductor Solutions segment was

Company Name: Broadcom | Quarter: Q4 | % year-on-year with Infrastructure Software operating margin at 75%. Now moving on to cash flow. Free cash flow in the quarter was $4.7 billion and represented 51% of revenues in Q4. We spent $105 million on capital expenditures. Days sales outstanding were 31 days in the fourth quarter compared to 30 days in the third. We ended the fourth quarter with inventory of $1.9 billion, up 3% sequentially. We continue to remain disciplined on how we manage inventory across the ecosystem. We exited the quarter with 76 days of inventory on hand, down 80 days in Q3. We ended the fourth quarter with $14.2 billion of cash and $39.2 billion of gross debt, of which $1.6 billion is short term. Now let me recap our financial performance for fiscal 2023. Our revenue had a record $35.8 billion, growing 8% year-on-year. Semiconductor revenue was $28.2

Company Name: Broadcom | Quarter: Q4 | segments, starting with our semiconductor segment. Revenue for our Semiconductor Solutions segment was $7.3 billion and represented 79% of total revenue in the quarter. This was up 3% year-on-year. Gross margins for our Semiconductor Solutions segment were approximately 70%, down 110 basis points year-on-year driven primarily by product mix within our semiconductor end markets. Operating expenses were stable year-on-year at $822 million, resulting in operating profit growth of 2% year-on-year and semiconductor operating margins of 58%. Now moving on to our Infrastructure Software segment. Revenue for Infrastructure Software was $2 billion, up 7% year-on-year and represented 21% of revenue. Gross margins for Infrastructure Software were 92% in the quarter, and operating expenses were $339 million in the quarter. Q4 operating profit grew 12% year-on-year with Infrastructure Software operating margin at 75%. Now moving on to cash flow

Company Name: Broadcom | Quarter: Q4 | Hock E. Tan President and Chief Executive Officer at Broadcom Thank you, Ji, and thank you, everyone, for joining us today. In our fiscal Q4 '23, consolidated net revenue was $9.3 billion, up 4% year-on-year and very much as we had guided at the last conference call. Semiconductor Solutions revenue increased 30% [Phonetic] year-on-year to $7.3 billion, and Infrastructure Software revenue grew 7% year-on-year to $2 billion. Overall, while Infrastructure Software remains very stable, semiconductor is continuing the cyclical slowdown at enterprises and telcos that we have been seeing over the past six months. However, hyperscalers remain strong. Generative AI revenue driven by Ethernet solutions and custom AI accelerators represented close to $1.5 billion in Q4 or 20% of semiconductor revenue, while the rest of the semiconductor revenue continued to be rather stable at

Company Name: Broadcom | Quarter: Q4 | billion, growing 8% year-on-year. Semiconductor revenue was $28.2 billion, up 9% year-over-year. Infrastructure Software revenue was $7.6 billion, up 3% year-on-year. Gross margin for the year was 74.7%, down 90 basis points from a year ago. Operating expenses were $4.6 billion, down 4% year-on-year. Fiscal 2023 operating income was $22.1 billion, up 9% year-over-year and represented 62% of net revenue. Adjusted EBITDA was $23.2 billion, up 10% year-over-year and represented 65% of net revenue. This figure excludes $502 million of depreciation. We spent $452 million on capital expenditures, and free cash flow grew 8% year-on-year to $17.6 billion or 49% of fiscal 2023 revenue. Now turning to capital allocation. For fiscal 2023

Company Name: Broadcom | Quarter: Q4 | as our VMware spending run rate exit fiscal '24 at approximately $1.4 billion per quarter, down 40% from a year ago. So in fiscal year 2024, including VMware, we expect consolidated adjusted EBITDA of approximately 60% of projected revenue. That concludes my prepared remarks. Operator, please open up the call for questions.

Company Name: Broadcom | Quarter: Q4 | % of semiconductor revenue, while the rest of the semiconductor revenue continued to be rather stable at around $6 billion. Moving on to results for the year. For fiscal 2023, consolidated revenue hit a record $35.8 billion, growing 8% year-on-year. And since 2020, even though we have not made an acquisition, we have shown a robust trajectory of growth driven by semiconductor growing at an 18% CAGR over the past three years. In fiscal 2023, operating profit grew by 9% year-on-year, and our free cash flow grew 8% year-on-year to $17.6 billion or 49% of revenue. We returned $13.5 billion in cash to our shareholders through dividends and stock buybacks. As you well know, we just closed the acquisition of VMware on November 22, just about four weeks into Broadcom's fiscal 2024. We are now refocusing VMware on its core business of creating private

Company Name: Broadcom | Quarter: Q4 | onductor revenue. In fiscal '23, wireless revenue was relatively flat at $7.3 billion in fact, just down 2% year-on-year. The engagement with our North American customers continues to be deep, strategic and multiyear. And accordingly, in fiscal '24, we expect wireless revenue to again remain stable year-on-year. Next, our Q4 server storage connected -- connectivity revenue was $1 billion or 14% of semiconductor revenue and down 17% year-on-year. In fiscal '23, server storage connectivity was $4.5 billion, up 11% year-on-year. And going to fiscal '24, we expect server storage revenue to decline mid- to high teens percentage year-on-year, driven by the cyclical weakness that began late '23. And moving on to broadband, Q4 revenue declined 9% year-on-year to $950 million, in line with expectations and represented 13% of semiconductor

Question: What was Broadcom's revenue in Q4 and how did it compare to the previous quarter?""",
        output="Broadcom's revenue in Q4 was $9.3 billion, which was up 4% from the previous quarter.",
        model="gpt-4",
        name="GPT Chat",
        num_input_tokens=50,
        num_output_tokens=20,
        total_tokens=70,
        duration_ns=int(1.2e8),
        metadata={"temperature": "0.7", "model_version": "1.0"},
        temperature=0.7,
        status_code=200,
        time_to_first_token_ns=500000,
    )

    # Conclude the workflow span
    print("Concluding workflow span...")
    logger.conclude(
        output='Broadcom\'s revenue in Q4 was $9.3 billion, which was up 4% from the previous quarter.',
        duration_ns=int(2.5e8),
        status_code=200
    )
    logger.flush()

if __name__ == "__main__":
    main()