from __future__ import annotations
import ast
import json
from datetime import datetime, timezone
from typing import Any, Callable

class FunctionRegistry:
    def __init__(self): self._functions:dict[str,tuple[dict,Callable[...,Any]]]={}; self.register("calculator","Evaluate a basic arithmetic expression using safe operators.",{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]},self._calculate); self.register("current_time","Return current UTC time.",{"type":"object","properties":{}},self._current_time)
    def register(self,name:str,description:str,parameters:dict,handler:Callable[...,Any])->None:
        if not name.replace("_","").isalnum():raise ValueError("function name must be alphanumeric with underscores")
        self._functions[name]=({"name":name,"description":description,"parameters":parameters},handler)
    def schemas(self)->list[dict]:return [spec for spec,_ in self._functions.values()]
    def call(self,name:str,arguments:dict)->Any:
        if name not in self._functions:raise ValueError(f"Unknown function: {name}")
        return self._functions[name][1](**arguments)
    @staticmethod
    def _calculate(expression:str)->dict:
        allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,ast.USub,ast.UAdd)
        try: tree=ast.parse(expression,mode="eval")
        except SyntaxError as error:raise ValueError("Invalid arithmetic expression") from error
        if not all(isinstance(node,allowed) and not (isinstance(node,ast.Constant) and not isinstance(node.value,(int,float))) for node in ast.walk(tree)):raise ValueError("Expression contains an unsupported operation")
        value=eval(compile(tree,"<calculator>","eval"),{"__builtins__":{}},{})
        if not isinstance(value,(int,float)) or abs(value)>1e100:raise ValueError("Result is not a finite supported number")
        return {"expression":expression,"result":value}
    @staticmethod
    def _current_time()->dict:return {"utc":datetime.now(timezone.utc).isoformat()}

def parse_tool_call(text:str)->tuple[str,dict[str,Any]]|None:
    start=text.find("<tool>"); end=text.find("</tool>",start)
    if start<0 or end<0:return None
    try:
        obj=json.loads(text[start+6:end].strip()); name=obj["name"]; arguments=obj.get("arguments",{})
        if isinstance(name,str) and isinstance(arguments,dict):return name,arguments
    except (json.JSONDecodeError,KeyError,TypeError):return None
    return None
