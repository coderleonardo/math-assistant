from typing import List

from pydantic import BaseModel, Field


class ResolutionStep(BaseModel):
    """One step of a solution's Chain of Thought, as extracted by an LLM."""

    step_number: int = Field(description="The step's position in the logical solution sequence.")
    description: str = Field(description="Short textual explanation of what happens in this step.")
    formula_latex: str = Field(
        description="The main equation for this step, in LaTeX. Leave empty if there is none."
    )


class Concept(BaseModel):
    """A mathematical concept used to solve a problem, with disambiguation context.

    The description is what makes the later entity-resolution pass work: two concepts
    that share a name but mean different things must not be merged, and two differently
    phrased names for the same idea must be. See enrichment.concept_resolution.
    """

    name: str = Field(
        description=(
            "The concept's standard mathematical name "
            "(e.g. Line Integral, Cayley-Hamilton Theorem)."
        )
    )
    description: str = Field(
        description=(
            "One sentence, grounded in this specific problem, explaining how the concept "
            "is used here. Used later to disambiguate concepts with similar names."
        )
    )


class GraphExtraction(BaseModel):
    """The structured-output shape requested from extract_graph_entities[_groq]."""

    concepts: List[Concept] = Field(
        description="Up to 5 specific mathematical concepts used to solve the problem."
    )
    resolution_steps: List[ResolutionStep] = Field(
        description="The accepted answer broken down into sequential logical steps (Chain of Thought)."
    )


class ConceptCluster(BaseModel):
    """A group of concept names that all refer to the same mathematical idea."""

    canonical: str = Field(
        description="The most standard, unambiguous mathematical name for this concept."
    )
    aliases: List[str] = Field(
        description=(
            "Every input name belonging to this cluster, including the canonical one. "
            "A genuinely distinct concept forms a single-element cluster."
        )
    )


class ResolvedConcepts(BaseModel):
    """The structured-output shape requested from resolve_concepts_gemini/_groq."""

    clusters: List[ConceptCluster] = Field(
        description=(
            "The input concepts partitioned into clusters. Each input name appears exactly once."
        )
    )
