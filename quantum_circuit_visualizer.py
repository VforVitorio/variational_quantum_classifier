import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# 1. Definir los símbolos matemáticos para que salgan en el dibujo
x = Parameter('x')
y = Parameter('y')
theta = [Parameter(f'θ_{i}') for i in range(1, 9)]  # Crea θ_1 hasta θ_8

# 2. Crear circuito (2 qubits, 2 bits clásicos)
qc = QuantumCircuit(2, 2)

# --- Encoding ---
qc.rx(x, 0)
qc.ry(y, 1)
qc.barrier()

# --- Layer 1 ---
qc.ry(theta[0], 0)  # theta_1
qc.ry(theta[1], 1)  # theta_2
qc.cx(0, 1)        # CORREGIDO: .cx en lugar de .cnot
qc.rx(theta[2], 0)  # theta_3
qc.rx(theta[3], 1)  # theta_4
qc.barrier()

# --- Layer 2 ---
qc.ry(theta[4], 0)  # theta_5
qc.ry(theta[5], 1)  # theta_6
qc.cx(0, 1)        # CORREGIDO: .cx en lugar de .cnot
qc.rx(theta[6], 0)  # theta_7
qc.rx(theta[7], 1)  # theta_8
qc.barrier()

# --- Measurement ---
qc.measure([0, 1], [0, 1])

# 3. Dibujar y guardar
# style='iqp' le da ese look moderno (High Contrast)
# scale=1.2 hace que el texto sea más grande y legible
fig = qc.draw(output='mpl', style='iqp', scale=1.2)

# Guardar imagen
filename = "results/circuito_vqc_final.png"
fig.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✅ Imagen generada correctamente: {filename}")
