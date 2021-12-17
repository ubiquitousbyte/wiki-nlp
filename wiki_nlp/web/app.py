from fastapi import (
    FastAPI,
    Depends
)
from multiprocessing import set_start_method


# This is needed because the data serialization utilities defined by PyTorch
# are not compatible with uvicorn.
# We need to import the objects to be serialized before uvicorn runs
# so that they can get included in uvicorn's main module list.
# This is one of the nastiest side effects I've encountered.
from wiki_nlp.data.dataset import (
    WikiDataset,
    WikiExample
)
import uvicorn
from wiki_nlp.web import (
    document_router,
)

app = FastAPI(title='Plagiarism Detector', version='0.1.0')

app.include_router(
    document_router.router,
    prefix='/api/v1'
)

if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    uvicorn.run(app, host='0.0.0.0', port=8000,
                log_level='debug')
