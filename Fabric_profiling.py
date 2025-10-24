
import os
import json
import base64
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import google.generativeai as genai
from PIL import Image
import numpy as np
from datetime import datetime


@dataclass
class FabricAnalysisResult:
    """Structured result from Gemini fabric analysis"""
    fabric_type: str
    confidence: float
    weave_pattern: str
    fiber_composition: List[str]
    texture_description: str
    crimp_observations: Dict[str, Any]
    quality_assessment: str
    color_description: str
    suggested_thread_count: Optional[str]
    defects_detected: List[str]
    timestamp: str
    model_used: str


class GeminiFabricRecognizer:
    """
    Fabric recognition using Google Gemini Vision API
    Provides detailed fabric analysis for fingerprinting system
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini fabric recognizer
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use (gemini-2.0-flash-exp, gemini-1.5-pro, etc.)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config for consistent JSON responses
        self.generation_config = {
            "temperature": 0.2,  # Lower temperature for more consistent technical analysis
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Safety settings (permissive for technical content)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (Gemini limit: 20MB, 4096x4096)
        max_size = 3072
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return img
    
    def _create_analysis_prompt(self, analysis_type: str = "comprehensive") -> str:
        """
        Create specialized prompt for fabric analysis
        
        Args:
            analysis_type: "comprehensive", "crimp", "type", or "quality"
        """
        base_prompt = """You are an expert textile engineer analyzing fabric samples for a nondestructive fingerprinting system.

Analyze this fabric image and provide detailed technical information in JSON format."""
        
        if analysis_type == "comprehensive":
            prompt = base_prompt + """

Return a JSON object with these fields:
{
  "fabric_type": "specific fabric type (e.g., cotton plain weave, polyester twill, wool jersey)",
  "confidence": 0.0-1.0 (your confidence in the identification),
  "weave_pattern": "plain/twill/satin/knit/other - describe the structure",
  "fiber_composition": ["list", "of", "likely", "fibers"],
  "texture_description": "detailed surface texture observations",
  "crimp_observations": {
    "visible_crimp": true/false,
    "crimp_regularity": "uniform/variable/irregular",
    "warp_crimp_visible": true/false,
    "weft_crimp_visible": true/false,
    "estimated_crimp_level": "low/medium/high",
    "yarn_interlacing": "description of how yarns cross"
  },
  "quality_assessment": "overall quality observations",
  "color_description": "detailed color and any patterns",
  "suggested_thread_count": "if visible, estimate threads per inch or 'not determinable'",
  "defects_detected": ["list", "any", "visible", "defects"]
}

Focus on technical accuracy. If uncertain about any field, indicate lower confidence."""
        
        elif analysis_type == "crimp":
            prompt = base_prompt + """

Focus specifically on crimp analysis. Return JSON:
{
  "crimp_visible": true/false,
  "warp_direction_crimp": "description and estimated percentage if visible",
  "weft_direction_crimp": "description and estimated percentage if visible",
  "crimp_amplitude": "low/medium/high - how pronounced are the waves",
  "crimp_wavelength": "tight/medium/loose spacing between crimps",
  "yarn_interlacing_description": "how yarns cross and bend",
  "recommended_measurement_approach": "suggestions for image processing"
}"""
        
        elif analysis_type == "type":
            prompt = base_prompt + """

Focus on fabric type classification. Return JSON:
{
  "primary_classification": "the main fabric type",
  "secondary_classifications": ["alternative", "possibilities"],
  "confidence_scores": {"type1": 0.8, "type2": 0.15, "type3": 0.05},
  "weave_structure": "detailed weave description",
  "fiber_type_indicators": ["visual clues suggesting fiber type"],
  "similar_fabrics": ["fabrics with similar appearance"]
}"""
        
        elif analysis_type == "quality":
            prompt = base_prompt + """

Focus on quality assessment. Return JSON:
{
  "overall_quality": "poor/fair/good/excellent",
  "uniformity_score": 0.0-1.0,
  "defects": ["list", "of", "any", "defects"],
  "yarn_evenness": "description",
  "surface_regularity": "description",
  "recommended_uses": ["suggested applications based on quality"]
}"""
        
        return prompt
    
    def analyze_fabric(
        self,
        image_path: str,
        analysis_type: str = "comprehensive",
        scale_mm_per_px: Optional[float] = None
    ) -> FabricAnalysisResult:
        """
        Analyze fabric image using Gemini Vision API
        
        Args:
            image_path: Path to fabric image
            analysis_type: Type of analysis to perform
            scale_mm_per_px: Optional calibration scale for dimensional analysis
        
        Returns:
            FabricAnalysisResult with structured data
        """
        img = self._load_image(image_path)
        prompt = self._create_analysis_prompt(analysis_type)
        
        # Add scale information if provided
        if scale_mm_per_px:
            prompt += f"\n\nNote: Image scale is {scale_mm_per_px:.4f} mm per pixel. "
            prompt += "Use this for dimensional estimates if relevant."
        
        try:
            response = self.model.generate_content(
                [prompt, img],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Parse JSON response
            response_text = response.text
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result_dict = json.loads(response_text)
            
            # Create structured result
            result = FabricAnalysisResult(
                fabric_type=result_dict.get("fabric_type", "unknown"),
                confidence=float(result_dict.get("confidence", 0.0)),
                weave_pattern=result_dict.get("weave_pattern", "unknown"),
                fiber_composition=result_dict.get("fiber_composition", []),
                texture_description=result_dict.get("texture_description", ""),
                crimp_observations=result_dict.get("crimp_observations", {}),
                quality_assessment=result_dict.get("quality_assessment", ""),
                color_description=result_dict.get("color_description", ""),
                suggested_thread_count=result_dict.get("suggested_thread_count"),
                defects_detected=result_dict.get("defects_detected", []),
                timestamp=datetime.now().isoformat(),
                model_used=self.model_name
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Gemini API analysis failed: {str(e)}")
    
    def batch_analyze(
        self,
        image_paths: List[str],
        analysis_type: str = "comprehensive",
        output_dir: Optional[str] = None
    ) -> List[FabricAnalysisResult]:
        """
        Analyze multiple fabric images in batch
        
        Args:
            image_paths: List of image paths
            analysis_type: Type of analysis
            output_dir: Optional directory to save results
        
        Returns:
            List of FabricAnalysisResult objects
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.analyze_fabric(image_path, analysis_type)
                results.append(result)
                
                # Save individual result if output directory specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = Path(image_path).stem
                    output_path = os.path.join(output_dir, f"{filename}_analysis.json")
                    
                    with open(output_path, 'w') as f:
                        json.dump(asdict(result), f, indent=2)
                    
                    print(f"  ✓ Saved to {output_path}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue
        
        return results
    
    def compare_fabrics(
        self,
        image_path1: str,
        image_path2: str
    ) -> Dict[str, Any]:
        """
        Compare two fabric samples using Gemini
        
        Args:
            image_path1: Path to first fabric image
            image_path2: Path to second fabric image
        
        Returns:
            Comparison analysis dictionary
        """
        img1 = self._load_image(image_path1)
        img2 = self._load_image(image_path2)
        
        prompt = """You are comparing two fabric samples. Analyze both images and provide a detailed comparison.

Return JSON:
{
  "overall_similarity": 0.0-1.0,
  "fabric_type_match": true/false,
  "differences": ["list", "of", "key", "differences"],
  "similarities": ["list", "of", "similarities"],
  "weave_comparison": "comparison of weave patterns",
  "color_comparison": "comparison of colors",
  "quality_comparison": "which appears higher quality and why",
  "replication_feasibility": "assessment of how easily fabric1 could replicate fabric2"
}"""
        
        try:
            response = self.model.generate_content(
                [prompt, img1, img2],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            raise RuntimeError(f"Comparison failed: {str(e)}")
    
    def get_crimp_guidance(self, image_path: str) -> Dict[str, Any]:
        """
        Get specific guidance for crimp measurement from image
        
        Args:
            image_path: Path to fabric image
        
        Returns:
            Dictionary with crimp measurement guidance
        """
        img = self._load_image(image_path)
        
        prompt = """Analyze this fabric image to provide guidance for automated crimp percentage measurement.

Return JSON:
{
  "crimp_visibility": "clear/partial/poor - how visible is the crimp",
  "best_measurement_region": "description of where to focus analysis",
  "recommended_preprocessing": ["steps", "for", "image processing"],
  "warp_direction_estimate": "horizontal/vertical/diagonal - which way runs warp",
  "weft_direction_estimate": "perpendicular to warp",
  "expected_crimp_range": "rough estimate like '5-15%'",
  "challenges": ["potential", "issues", "for", "automated analysis"],
  "yarn_visibility_score": 0.0-1.0,
  "suggested_magnification": "current magnification appears adequate/insufficient"
}"""
        
        try:
            response = self.model.generate_content(
                [prompt, img],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            raise RuntimeError(f"Crimp guidance failed: {str(e)}")


# Integration with main fingerprinting system
def integrate_with_fingerprint(
    gemini_result: FabricAnalysisResult,
    crimp_data: Dict[str, float],
    thickness_data: Dict[str, float],
    ml_prediction: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine Gemini analysis with other fingerprinting data
    
    Args:
        gemini_result: Result from Gemini analysis
        crimp_data: Crimp percentage measurements
        thickness_data: Thickness measurements
        ml_prediction: ML model predictions
    
    Returns:
        Enhanced fingerprint with Gemini insights
    """
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "gemini_model": gemini_result.model_used
        },
        "measurements": {
            "crimp": crimp_data,
            "thickness_mm": thickness_data
        },
        "ml_classification": ml_prediction,
        "gemini_analysis": {
            "fabric_type": gemini_result.fabric_type,
            "confidence": gemini_result.confidence,
            "weave_pattern": gemini_result.weave_pattern,
            "fiber_composition": gemini_result.fiber_composition,
            "texture": gemini_result.texture_description,
            "crimp_observations": gemini_result.crimp_observations,
            "quality": gemini_result.quality_assessment,
            "defects": gemini_result.defects_detected
        },
        "combined_assessment": {
            "fabric_type_agreement": ml_prediction.get("label") == gemini_result.fabric_type.split()[0],
            "confidence_score": (gemini_result.confidence + ml_prediction.get("confidence", 0)) / 2,
            "recommended_label": gemini_result.fabric_type
        }
    }


# CLI interface
def main():
    """Command-line interface for Gemini fabric recognition"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fabric Recognition using Google Gemini API")
    parser.add_argument("--image", required=True, help="Path to fabric image")
    parser.add_argument("--api-key", help="Google AI API key (or set GOOGLE_API_KEY env)")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", help="Gemini model name")
    parser.add_argument("--analysis-type", default="comprehensive", 
                       choices=["comprehensive", "crimp", "type", "quality"],
                       help="Type of analysis to perform")
    parser.add_argument("--scale", type=float, help="Scale in mm per pixel")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--batch", nargs="+", help="Batch process multiple images")
    parser.add_argument("--batch-output-dir", help="Directory for batch results")
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = GeminiFabricRecognizer(api_key=args.api_key, model_name=args.model)
    
    if args.batch:
        # Batch processing
        print(f"\n🔍 Batch analyzing {len(args.batch)} images...\n")
        results = recognizer.batch_analyze(
            args.batch,
            analysis_type=args.analysis_type,
            output_dir=args.batch_output_dir
        )
        print(f"\n✅ Completed {len(results)}/{len(args.batch)} analyses")
        
    else:
        # Single image analysis
        print(f"\n🔍 Analyzing: {args.image}")
        print(f"📊 Analysis type: {args.analysis_type}")
        if args.scale:
            print(f"📏 Scale: {args.scale:.4f} mm/px")
        
        result = recognizer.analyze_fabric(
            args.image,
            analysis_type=args.analysis_type,
            scale_mm_per_px=args.scale
        )
        
        # Display results
        print(f"\n✅ Analysis complete!")
        print(f"\n📋 Fabric Type: {result.fabric_type}")
        print(f"🎯 Confidence: {result.confidence:.2%}")
        print(f"🔗 Weave Pattern: {result.weave_pattern}")
        print(f"🧵 Fiber Composition: {', '.join(result.fiber_composition)}")
        print(f"\n📝 Texture: {result.texture_description}")
        print(f"\n⭐ Quality: {result.quality_assessment}")
        
        if result.crimp_observations:
            print(f"\n📐 Crimp Observations:")
            for key, value in result.crimp_observations.items():
                print(f"  • {key}: {value}")
        
        if result.defects_detected:
            print(f"\n⚠️  Defects Detected: {', '.join(result.defects_detected)}")
        
        # Save to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()