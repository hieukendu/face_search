# app.py

from fastapi import FastAPI, UploadFile


app = FastAPI()


@app.get("/health")
def health():
    """
    Test server hoạt động.
    """
    return {"status": "ok"}


@app.post("/search")
async def search_photo(file: UploadFile):
    """
    User upload selfie.

    Pipeline:

        uploaded image
        -> SearchEngine.search()
        -> matching photos

    Trả JSON cho frontend.
    """
    pass