import gradio as gr
from rag_pipeline import query, classify_query, DEMO_QUESTIONS

# Theme & Styling

CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.main-title h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.05em;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
}
.stats-row {
    display: flex;
    gap: 1em;
    justify-content: center;
    margin-bottom: 1em;
}
.stat-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 12px 20px;
    text-align: center;
    min-width: 140px;
}
.stat-card .label {
    font-size: 0.8em;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stat-card .value {
    font-size: 1.4em;
    font-weight: 700;
    color: #4338ca;
}
footer { display: none; }
"""

EXAMPLE_QUERIES = [
    "What is the sales trend over the 4-year period?",
    "Which months show the highest sales? Is there seasonality?",
    "Which product category generates the most revenue?",
    "What sub-categories have the highest profit margins?",
    "Which region has the best sales performance?",
    "Which cities are the top performers?",
    "Compare Technology vs. Furniture sales trends.",
    "How does the West region compare to the East in profit?",
]


# Query Handler


def handle_query(question: str, history: list) -> tuple:
    # Process a question through the RAG pipeline and return formatted output.
    if not question.strip():
        return history, "", "", ""

    # Add user message to chat
    history = history + [{"role": "user", "content": question}]

    try:
        result = query(question, verbose=False)

        answer = result["answer"]
        query_types = ", ".join(result["query_types"])
        timing = (
            f"⏱ Retrieval: {result['retrieval_time']}s · "
            f"Generation: {result['generation_time']}s · "
            f"Total: {result['total_time']}s"
        )
        meta = f" Query type: {query_types}  |   Context: {result['context_length']} chars"

        # Add assistant response to chat
        history = history + [{"role": "assistant", "content": answer}]

        return history, "", timing, meta

    except Exception as e:
        error_msg = f" Error: {str(e)}\n\nMake sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull mistral`)."
        history = history + [{"role": "assistant", "content": error_msg}]
        return history, "", "", ""


def set_example(example: str) -> str:
    """Set an example query in the input box."""
    return example

# Build the UI

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="RAG Sales Analyst",
    ) as app:

        # Header
        gr.HTML("""
            <div class="main-title"><h1> RAG Sales Analyst</h1></div>
            <div class="subtitle">
                Retrieval-Augmented Generation for Superstore Sales Data (2014–2017)
            </div>
            <div class="stats-row">
                <div class="stat-card">
                    <div class="label">Transactions</div>
                    <div class="value">9,994</div>
                </div>
                <div class="stat-card">
                    <div class="label">Documents</div>
                    <div class="value">5,168</div>
                </div>
                <div class="stat-card">
                    <div class="label">Embedding</div>
                    <div class="value">MiniLM</div>
                </div>
                <div class="stat-card">
                    <div class="label">LLM</div>
                    <div class="value">Mistral 7B</div>
                </div>
            </div>
        """)

        # Chat area
        chatbot = gr.Chatbot(
            label="Conversation",
            height=420,
            avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/bar-chart_1f4ca.png"),
        )

        # Input row
        with gr.Row():
            question_input = gr.Textbox(
                placeholder="Ask a question about the sales data...",
                label="Your Question",
                scale=5,
                lines=1,
            )
            submit_btn = gr.Button("Ask", variant="primary", scale=1)

        # Metadata row
        with gr.Row():
            meta_display = gr.Textbox(label="Query Info", interactive=False, scale=3)
            timing_display = gr.Textbox(label="Performance", interactive=False, scale=2)

        # Example queries
        gr.Markdown("### Try these questions")
        with gr.Row(equal_height=True):
            with gr.Column():
                for eq in EXAMPLE_QUERIES[:4]:
                    btn = gr.Button(eq, size="sm", variant="secondary")
                    btn.click(fn=set_example, inputs=[btn], outputs=[question_input])
            with gr.Column():
                for eq in EXAMPLE_QUERIES[4:]:
                    btn = gr.Button(eq, size="sm", variant="secondary")
                    btn.click(fn=set_example, inputs=[btn], outputs=[question_input])

        # Clear button
        clear_btn = gr.Button("🗑 Clear Chat", size="sm")
        clear_btn.click(
            fn=lambda: ([], "", "", ""),
            outputs=[chatbot, question_input, timing_display, meta_display],
        )

        # Wire up submit
        submit_args = dict(
            fn=handle_query,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, timing_display, meta_display],
        )
        submit_btn.click(**submit_args)
        question_input.submit(**submit_args)

    return app

# Launch

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
