"""
Context-aware retriever with context window expansion.
Retrieves top-k documents and expands context by including adjacent chunks (before/after).
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document


class ContextWindowRetriever:
    """
    Enveloppe un récupérateur et étend le contexte en :
    1. Récupérant les top-k documents
    2. Pour chaque document, incluant le contexte environnant (chunks avant/après de la même source)
    3. Fusionnant et dédupliquant pour les résultats finaux
    """
    
    def __init__(self, vectorstore, top_k: int = 3, context_window: int = 1):
        """
        Args:
            vectorstore: Le vectorstore FAISS
            top_k: Nombre de résultats principaux à retourner (avant expansion)
            context_window: Nombre de chunks à inclure avant/après chaque résultat (1-2 recommandé)
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.context_window = context_window
        self.base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k}
        )
        # Construire une carte des documents par source pour l'expansion du contexte
        self._build_document_map()
    
    def _build_document_map(self):
        """Build a map of documents organized by source for context expansion."""
        self.source_documents: Dict[str, List[Document]] = {}
        
        # Get all documents from the vectorstore
        try:
            # Try to get all documents if available
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                all_docs = list(self.vectorstore.docstore._dict.values())
                
                # Group by source
                for doc in all_docs:
                    source = doc.metadata.get('source', 'unknown')
                    if source not in self.source_documents:
                        self.source_documents[source] = []
                    self.source_documents[source].append(doc)
                
                # Sort by page number if available
                for source in self.source_documents:
                    try:
                        self.source_documents[source].sort(
                            key=lambda x: (x.metadata.get('page', 0), x.metadata.get('chunk_index', 0))
                        )
                    except:
                        pass
        except:
            pass
    
    def _find_document_index(self, doc: Document, source: str) -> int:
        """Find the index of a document in its source group."""
        if source not in self.source_documents:
            return -1
        
        for i, d in enumerate(self.source_documents[source]):
            if d.page_content == doc.page_content:
                return i
        return -1
    
    def _get_context_window(self, doc: Document) -> List[Document]:
        """Get the document and its surrounding context (before/after)."""
        source = doc.metadata.get('source', 'unknown')
        
        if source not in self.source_documents:
            return [doc]
        
        idx = self._find_document_index(doc, source)
        if idx == -1:
            return [doc]
        
        # Get window of documents
        start_idx = max(0, idx - self.context_window)
        end_idx = min(len(self.source_documents[source]), idx + self.context_window + 1)
        
        return self.source_documents[source][start_idx:end_idx]
    
    def _merge_documents(self, docs: List[Document]) -> Document:
        """Merge multiple documents into one with better separator and preserve metadata."""
        if not docs:
            return None
        
        # Use main document's metadata
        main_doc = docs[0]
        
        # Merge content with clear separator
        merged_content = "\n\n---\n\n".join([d.page_content for d in docs])
        
        # Preserve the dl_meta structure from the main document
        metadata = {**main_doc.metadata}
        
        # Preserve dl_meta if it exists
        if 'dl_meta' in main_doc.metadata:
            dl_meta = main_doc.metadata['dl_meta']
            # Update doc_items to include all pages from merged documents
            if isinstance(dl_meta, dict) and 'doc_items' in dl_meta:
                # Collect all doc_items from all merged documents
                all_doc_items = []
                for doc in docs:
                    if 'dl_meta' in doc.metadata and isinstance(doc.metadata['dl_meta'], dict):
                        doc_items = doc.metadata['dl_meta'].get('doc_items', [])
                        if isinstance(doc_items, list):
                            all_doc_items.extend(doc_items)
                
                # Update metadata with all items
                if all_doc_items:
                    metadata['dl_meta'] = {**dl_meta, 'doc_items': all_doc_items}
        
        merged_doc = Document(
            page_content=merged_content,
            metadata={
                **metadata,
                'expanded_from': len(docs),
                'expansion': 'context_window',
                'original_pages': [d.metadata.get('page', 'unknown') for d in docs]
            }
        )
        
        return merged_doc
    
    def invoke(self, query: str) -> List[Document]:
        """
        Retrieve documents and expand context around each result.
        
        Args:
            query: The search query
            
        Returns:
            List of documents with expanded context
        """
        # Get initial top-k results
        top_docs = self.base_retriever.invoke(query)
        
        if not top_docs:
            return top_docs
        
        # Expand context around each result
        expanded_docs = []
        processed_sources = set()
        
        for doc in top_docs:
            source = doc.metadata.get('source', 'unknown')
            
            # Get surrounding documents
            context_docs = self._get_context_window(doc)
            
            # Merge them
            merged = self._merge_documents(context_docs)
            
            # Avoid duplicates (same source already processed)
            if source not in processed_sources:
                expanded_docs.append(merged)
                processed_sources.add(source)
        
        return expanded_docs


def create_enhanced_retriever(vectorstore, top_k: int = 3, context_window: int = 1):
    """
    Create a retriever with context window expansion.
    
    Args:
        vectorstore: The FAISS vectorstore
        top_k: Number of top results to return
        context_window: Number of adjacent chunks to include (1-2 recommended)
        
    Returns:
        A ContextWindowRetriever instance
    """
    return ContextWindowRetriever(vectorstore, top_k=top_k, context_window=context_window)


