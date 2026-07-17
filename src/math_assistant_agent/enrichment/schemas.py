from typing import List

from pydantic import BaseModel, Field


class ResolutionStep(BaseModel):
    step_number: int = Field(description="The step's position in the logical solution sequence.")
    description: str = Field(description="Short textual explanation of what happens in this step.")
    formula_latex: str = Field(
        description="The main equation for this step, in LaTeX. Leave empty if there is none."
    )


class GraphExtraction(BaseModel):
    domain: str = Field(
        description="The broad area of mathematics (e.g. Calculus, Linear Algebra, Number Theory)."
    )
    concepts: List[str] = Field(
        description=(
            "Up to 5 specific mathematical concepts used to solve the problem "
            "(e.g. Line Integral, Cayley-Hamilton Theorem)."
        )
    )
    resolution_steps: List[ResolutionStep] = Field(
        description="The accepted answer broken down into sequential logical steps (Chain of Thought)."
    )
