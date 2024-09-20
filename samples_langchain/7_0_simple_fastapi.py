from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# run cml : uvicorn samples_langchain.7_0_simple_fastapi:app --reload
# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/quote
# distribute : cloudflared tunnel --url http://127.0.0.1:8000
# note : install cml for cloudflared is "winget install --id Cloudflare.cloudflared" refer to homepage 
app = FastAPI(
    title="Nicolacus Maximus Quote Giver",
    description="Get a real quote said by Nicolacus Maximus himself.",
    servers=[{"url":"https://bye-organize-closes-accomplish.trycloudflare.com"}],
)


class Quote(BaseModel):
    quote: str = Field(
        description="The quote that Nicolacus Maximus said.",
    )
    year: int = Field(
        description="The year when Nicolacus Maximus said the quote.",
    )


@app.get(
    "/quote",
    summary="Returns a random quote by Nicolacus Maximus",
    description="Upon receiving a GET request this endpoint will return a real quiote said by Nicolacus Maximus himself.",
    response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
    response_model=Quote,
)
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950,
    }