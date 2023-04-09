
from ..pipelines import CustomPipeline
from ._router import BaseRouter
from schemas.pipelines import PipelineDescription,PipelinesResponse,ComponentDescription,PrimitiveType
import haystack
from networkx.drawing.nx_agraph import to_agraph
import os

class PipelineRouter(BaseRouter):
    def __init__(self,search_pipeline:CustomPipeline,extractive_qa_pipeline:CustomPipeline):
        super().__init__("/pipeline")
        self.search_pipeline = search_pipeline
        self.extractive_qa_pipeline = extractive_qa_pipeline
        self.router.add_api_route("/pipelines", self.get_pipelines, methods=["GET"], response_model=PipelinesResponse)
        


    async def get_pipelines(self)-> PipelinesResponse:
        """
        Lists the pipelines of this node and generates a DOT-Graph for each pipeline
        """
        pipelines = PipelinesResponse(pipelines=[])
        hf_token = os.getenv("HUGGINGFACE_TOKEN",None)
        open_ai_token = os.getenv("OPENAI_TOKEN",None)
        
        def process_pipeline(name:str,pipeline:haystack.Pipeline)->PipelineDescription:
            components = []
            for component_name,component in pipeline.components.items():
                
                if "DocumentStore" in component_name:
                    continue
                
                componentDescription=ComponentDescription(name=component_name,params={})
                for key,value in component.get_params(return_defaults=True).items():
                    if isinstance(value,str):
                        #Don't leak the api keys
                        if hf_token and hf_token in value:
                            value = value.replace(hf_token,"HUGGINGFACE_TOKEN")
                        if open_ai_token and open_ai_token in value:
                            value = value.replace(open_ai_token,"OPENAI_TOKEN")
                            
                    if isinstance(value,PrimitiveType):
                        componentDescription.params[key]=value
                components.append(componentDescription)

            agraph = to_agraph(pipeline.graph)
            dot = agraph.to_string()
            
            return PipelineDescription(name=name,components=components,graph=dot)
        
        pipelines.pipelines.append(process_pipeline("search_pipeline",self.search_pipeline.pipeline))
        pipelines.pipelines.append(process_pipeline("extractive_qa_pipeline",self.extractive_qa_pipeline.pipeline))
        return pipelines

