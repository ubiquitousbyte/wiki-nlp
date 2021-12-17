from wiki_nlp.web import dependencies


if __name__ == '__main__':
    from fastapi import (
        FastAPI,
        Depends
    )
    from wiki_nlp.web import (
        document_router,
    )

    app = FastAPI(title='Plagiarism Detector', version='0.1.0')

    app.include_router(
        document_router.router,
        prefix='/api/v1'
    )
