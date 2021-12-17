from typing import (
    Dict,
    Tuple,
    List
)

from fastapi import (
    APIRouter,
    Depends
)

from wiki_nlp.data.domain import Document
from wiki_nlp.web import dependencies as di
from wiki_nlp.services import (
    dm_service,
    som_service
)

router = APIRouter(prefix='/docs')


@router.get('/som')
def get_som(
    som_service: som_service.ISOMService = Depends(di.get_som_document_service)
):
    m = som_service.get_map()
    print(type(m))
    return m


@router.post('/som', response_model=Tuple[int, int])
def predict(
    document: Document,
    som_service: som_service.ISOMService = Depends(
        di.get_som_document_service),
    dm_service: dm_service.IDMService = Depends(di.get_dm_document_service)
):
    vec = dm_service.infer_vector(document)
    winner = som_service.winner(vec)
    return winner


@router.post('/dm')
def most_similar(
    document: Document,
    dm_service: dm_service.IDMService = Depends(di.get_dm_document_service)
):
    vec = dm_service.infer_vector(document)
    dists, indices = dm_service.most_similar(vec, topn=1)
    results = [{'similarity': dist.item(), 'document': dm_service[indices[i].item()]}
               for i, dist in enumerate(dists)]
    return results
