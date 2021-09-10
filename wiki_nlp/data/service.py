from typing import Iterator 

from gql import (
    gql, 
    Client 
)
from gql.transport.aiohttp import AIOHTTPTransport

from wiki_nlp.data.domain import Document

class DocumentService:
    # The document service represents a GraphQL client that connects to our
    # ZIO backend and extracts document batches. 

    async def __aenter__(self):
        transport = AIOHTTPTransport(url='http://localhost:8080/api/graphql')
        self._client = Client(transport=transport, fetch_schema_from_transport=True)
        return self 

    async def __aexit__(self, exc_type, exc, tb):
        await self._client.__aexit__(exc_type, exc, tb)

    async def read_document_batch(self, offset: int = 0, limit: int = 10) -> Iterator[Document]:
        async with self._client as session:
            query = gql(
                """
                query($limit: Int!, $offset: Int!) {
                    documents(limit: $limit, offset: $offset) {
                        id
                        title
                        excerpt
                        source
                        sections {
                            id 
                            title
                            position
                            paragraphs {
                                id 
                                text
                            }
                        }
                    }
                }
                """
            )
            params = { 'offset': offset, 'limit': limit }
            result_set = await session.execute(query, variable_values=params)   
            for doc in result_set['documents']:
                yield Document(**doc)            