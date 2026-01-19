"""
Unreal Engine code generator for MechanicsDSL.

Generates C++ ActorComponent for Unreal Engine physics simulation.
"""

from typing import Any, Dict, List, Optional

try:
    import sympy as sp
    from sympy.printing import ccode
    SYMPY_AVAILABLE = True
except ImportError:
    sp = None
    ccode = None
    SYMPY_AVAILABLE = False


def sympy_to_cpp(expr: Any, use_fmath: bool = True) -> str:
    """
    Convert a sympy expression to C++ code for Unreal Engine.
    
    Args:
        expr: Sympy expression to convert
        use_fmath: If True, use FMath:: functions (Unreal). Otherwise use std::.
        
    Returns:
        C++ code string representing the expression
    """
    if expr is None:
        return "0.0f"
    
    if not SYMPY_AVAILABLE:
        return "0.0f /* sympy not available */"
    
    try:
        # Get C code representation
        c_code = ccode(expr)
        
        # Replace with Unreal Math functions if requested
        if use_fmath:
            replacements = [
                ('sin(', 'FMath::Sin('),
                ('cos(', 'FMath::Cos('),
                ('tan(', 'FMath::Tan('),
                ('asin(', 'FMath::Asin('),
                ('acos(', 'FMath::Acos('),
                ('atan(', 'FMath::Atan('),
                ('atan2(', 'FMath::Atan2('),
                ('sqrt(', 'FMath::Sqrt('),
                ('pow(', 'FMath::Pow('),
                ('exp(', 'FMath::Exp('),
                ('log(', 'FMath::Loge('),
                ('abs(', 'FMath::Abs('),
                ('fabs(', 'FMath::Abs('),
                ('floor(', 'FMath::FloorToFloat('),
                ('ceil(', 'FMath::CeilToFloat('),
                ('M_PI', 'PI'),
            ]
        else:
            replacements = [
                ('M_PI', 'PI'),
            ]
        
        for old, new in replacements:
            c_code = c_code.replace(old, new)
        
        # Ensure floating point literals with 'f' suffix
        import re
        c_code = re.sub(r'(\d+\.\d+)(?!f)', r'\1f', c_code)
        c_code = re.sub(r'(\d+)(?!\.\d)(?!f)(?![a-zA-Z_])', r'\1.0f', c_code)
        
        return c_code
        
    except Exception as e:
        return f"0.0f /* Error: {str(e)[:50]} */"


class UnrealGenerator:
    """
    Generate Unreal Engine C++ code from MechanicsDSL.

    Creates ActorComponent with physics simulation.

    Example:
        gen = UnrealGenerator(compiler)
        gen.generate('PendulumComponent.h', 'PendulumComponent.cpp')
    """

    @property
    def target_name(self) -> str:
        return "unreal"

    @property
    def file_extension(self) -> str:
        return ".h"

    def __init__(self, compiler=None):
        self.compiler = compiler
        self.equations: Dict[str, Any] = {}
        
        if compiler:
            self.system_name = getattr(compiler, "system_name", "Physics")
            self.coordinates: List[str] = getattr(compiler.simulator, "coordinates", [])
            self.parameters: Dict[str, float] = dict(getattr(compiler.simulator, "parameters", {}))
            self._extract_equations(compiler)
        else:
            self.system_name = "Physics"
            self.coordinates = []
            self.parameters = {}
    
    def _extract_equations(self, compiler) -> None:
        """
        Extract acceleration equations from the compiler.
        """
        # Try accelerations dict first
        if hasattr(compiler, 'accelerations') and compiler.accelerations:
            self.equations = dict(compiler.accelerations)
            return
            
        # Try simulator equations
        if hasattr(compiler, 'simulator') and hasattr(compiler.simulator, 'equations'):
            eqs = compiler.simulator.equations
            if isinstance(eqs, dict):
                self.equations = eqs
                return
            elif isinstance(eqs, list) and SYMPY_AVAILABLE:
                for i, eq in enumerate(eqs):
                    if i < len(self.coordinates):
                        self.equations[self.coordinates[i]] = eq
                return
        
        # Try equations_of_motion
        if hasattr(compiler, 'equations_of_motion') and compiler.equations_of_motion:
            eom = compiler.equations_of_motion
            if isinstance(eom, dict):
                self.equations = eom
            elif isinstance(eom, list) and SYMPY_AVAILABLE:
                for i, eq in enumerate(eom):
                    if i < len(self.coordinates):
                        self.equations[self.coordinates[i]] = eq

    def generate(self, header_file: Optional[str] = None, cpp_file: Optional[str] = None) -> tuple:
        """Generate Unreal C++ header and source files."""
        class_name = self._to_pascal_case(self.system_name)

        header = self._generate_header(class_name)
        source = self._generate_source(class_name)

        if header_file:
            with open(header_file, "w") as f:
                f.write(header)

        if cpp_file:
            with open(cpp_file, "w") as f:
                f.write(source)

        return header, source

    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))

    def _generate_header(self, class_name: str) -> str:
        """Generate header file."""
        return f"""// {class_name}Component.h
// Generated by MechanicsDSL

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "{class_name}Component.generated.h"

UCLASS(ClassGroup=(Physics), meta=(BlueprintSpawnableComponent))
class U{class_name}Component : public UActorComponent
{{
    GENERATED_BODY()

public:
    U{class_name}Component();

    // Physical Parameters
{self._generate_header_parameters()}

    // Simulation Settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
    float TimeScale = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
    bool bEnableSimulation = true;

    // Blueprint Callable Functions
    UFUNCTION(BlueprintCallable, Category = "Physics")
    void ResetSimulation();

    UFUNCTION(BlueprintCallable, Category = "Physics")
    TArray<float> GetState() const;

    UFUNCTION(BlueprintCallable, Category = "Physics")
    void SetState(const TArray<float>& NewState);

{self._generate_header_state_getters()}

protected:
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, 
                               FActorComponentTickFunction* ThisTickFunction) override;

private:
    // State array: [{', '.join(f'{c}, {c}_dot' for c in self.coordinates)}]
    TArray<float> State;
    
    void InitializeState();
    void RK4Step(float dt);
    void ComputeDerivatives(float t, const TArray<float>& y, TArray<float>& dydt);
    void UpdateOwnerTransform();
}};
"""

    def _generate_source(self, class_name: str) -> str:
        """Generate source file."""
        n_state = len(self.coordinates) * 2

        return f"""// {class_name}Component.cpp
// Generated by MechanicsDSL

#include "{class_name}Component.h"

U{class_name}Component::U{class_name}Component()
{{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup = TG_PrePhysics;
    
    State.SetNum({n_state});
}}

void U{class_name}Component::BeginPlay()
{{
    Super::BeginPlay();
    InitializeState();
}}

void U{class_name}Component::InitializeState()
{{
{self._generate_init_state()}
}}

void U{class_name}Component::TickComponent(float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    
    if (!bEnableSimulation) return;
    
    float dt = DeltaTime * TimeScale;
    RK4Step(dt);
    UpdateOwnerTransform();
}}

void U{class_name}Component::RK4Step(float dt)
{{
    const int32 n = State.Num();
    TArray<float> k1, k2, k3, k4, temp;
    k1.SetNum(n);
    k2.SetNum(n);
    k3.SetNum(n);
    k4.SetNum(n);
    temp.SetNum(n);
    
    float t = GetWorld()->GetTimeSeconds();
    
    // k1
    ComputeDerivatives(t, State, k1);
    
    // k2
    for (int32 i = 0; i < n; i++)
        temp[i] = State[i] + 0.5f * dt * k1[i];
    ComputeDerivatives(t + 0.5f * dt, temp, k2);
    
    // k3
    for (int32 i = 0; i < n; i++)
        temp[i] = State[i] + 0.5f * dt * k2[i];
    ComputeDerivatives(t + 0.5f * dt, temp, k3);
    
    // k4
    for (int32 i = 0; i < n; i++)
        temp[i] = State[i] + dt * k3[i];
    ComputeDerivatives(t + dt, temp, k4);
    
    // Update state
    for (int32 i = 0; i < n; i++)
        State[i] += dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0f;
}}

void U{class_name}Component::ComputeDerivatives(float t, const TArray<float>& y, TArray<float>& dydt)
{{
    // Unpack state
{self._generate_cpp_unpack()}

    // Compute accelerations (generated from equations)
{self._generate_cpp_accelerations()}

    // Pack derivatives
{self._generate_cpp_pack()}
}}

void U{class_name}Component::UpdateOwnerTransform()
{{
    if (AActor* Owner = GetOwner())
    {{
        // Example: Rotate based on first coordinate (assumed angle)
        float Angle = FMath::RadiansToDegrees(State[0]);
        Owner->SetActorRotation(FRotator(0.f, 0.f, -Angle));
    }}
}}

void U{class_name}Component::ResetSimulation()
{{
    InitializeState();
}}

TArray<float> U{class_name}Component::GetState() const
{{
    return State;
}}

void U{class_name}Component::SetState(const TArray<float>& NewState)
{{
    if (NewState.Num() == State.Num())
    {{
        State = NewState;
    }}
}}

{self._generate_cpp_getters()}
"""

    def _generate_header_parameters(self) -> str:
        """Generate UPROPERTY parameter declarations."""
        lines = []
        for name, value in self.parameters.items():
            lines.append(
                f'    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Parameters")'
            )
            lines.append(f"    float {name.capitalize()} = {value}f;")
            lines.append("")
        return "\n".join(lines) if lines else "    // No parameters"

    def _generate_header_state_getters(self) -> str:
        """Generate getter function declarations."""
        lines = []
        for coord in self.coordinates:
            lines.append(f'    UFUNCTION(BlueprintPure, Category = "State")')
            lines.append(f"    float Get{coord.capitalize()}() const;")
            lines.append("")
            lines.append(f'    UFUNCTION(BlueprintPure, Category = "State")')
            lines.append(f"    float Get{coord.capitalize()}Dot() const;")
            lines.append("")
        return "\n".join(lines)

    def _generate_init_state(self) -> str:
        """Generate state initialization."""
        lines = ["    State.Empty();"]
        for i, coord in enumerate(self.coordinates):
            lines.append(f"    State.Add(0.0f);  // {coord}")
            lines.append(f"    State.Add(0.0f);  // {coord}_dot")
        return "\n".join(lines)

    def _generate_cpp_unpack(self) -> str:
        """Generate C++ state unpacking."""
        lines = []
        for i, coord in enumerate(self.coordinates):
            lines.append(f"    float {coord} = y[{2*i}];")
            lines.append(f"    float {coord}_dot = y[{2*i + 1}];")
        return "\n".join(lines) if lines else "    // No state"

    def _generate_cpp_accelerations(self) -> str:
        """
        Generate C++ acceleration calculations from sympy equations.
        
        Converts symbolic equations to Unreal Engine C++ code using FMath functions.
        """
        lines = []
        
        for coord in self.coordinates:
            # Check if we have a real equation for this coordinate
            if coord in self.equations and self.equations[coord] is not None:
                accel_expr = self.equations[coord]
                cpp_code = sympy_to_cpp(accel_expr, use_fmath=True)
                lines.append(f"    float {coord}_ddot = {cpp_code};")
            elif f"{coord}_ddot" in self.equations:
                accel_expr = self.equations[f"{coord}_ddot"]
                cpp_code = sympy_to_cpp(accel_expr, use_fmath=True)
                lines.append(f"    float {coord}_ddot = {cpp_code};")
            else:
                lines.append(f"    float {coord}_ddot = 0.0f; // Warning: No equation for {coord}")
        
        return "\n".join(lines) if lines else "    // No accelerations defined"

    def _generate_cpp_pack(self) -> str:
        """Generate C++ derivative packing."""
        lines = []
        for i, coord in enumerate(self.coordinates):
            lines.append(f"    dydt[{2*i}] = {coord}_dot;")
            lines.append(f"    dydt[{2*i + 1}] = {coord}_ddot;")
        return "\n".join(lines) if lines else "    // No derivatives"

    def _generate_cpp_getters(self) -> str:
        """Generate C++ getter implementations."""
        lines = []
        for i, coord in enumerate(self.coordinates):
            class_name = self._to_pascal_case(self.system_name)
            lines.append(f"float U{class_name}Component::Get{coord.capitalize()}() const")
            lines.append("{")
            lines.append(f"    return State[{2*i}];")
            lines.append("}")
            lines.append("")
            lines.append(f"float U{class_name}Component::Get{coord.capitalize()}Dot() const")
            lines.append("{")
            lines.append(f"    return State[{2*i + 1}];")
            lines.append("}")
            lines.append("")
        return "\n".join(lines)


__all__ = ["UnrealGenerator"]
