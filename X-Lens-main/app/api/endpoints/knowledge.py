from fastapi import APIRouter, Request
router=APIRouter()
@router.get('/knowledge/status')
def knowledge_status(request:Request):
    rag=request.app.state.rag
    if not rag or not rag.retriever: return {"enabled":False,"chunks":0,"loaded":False}
    store=rag.retriever.store
    return {"enabled":True,"chunks":len(store.metadata),"loaded":store.index is not None,"threshold":rag.retriever.threshold}
