#include <iostream>
#include <cmath>
#include <vector>

struct GirosferaEstado {
    std::vector<double> posicion;
    std::vector<double> v; // Segundo momento (RMSprop / escala adaptativa) 
    std::vector<double> m; // primer momento(reemplaza a la velocidad clasica)
    std::vector<double> gradiente;

    double alpha = 0.01; // tasa de aprendizaje clasica
    double beta1 = 0.9; //factor de memoria
    double beta2 = 0.999; //factor de memoria para el segundo momento
    double epsilon = 1e-8; //Evita que el sistema explote en una division por 0
    int t = 0; //contado de pasos de tiempo

    // El constructor DEBE ir dentro del struct
    GirosferaEstado(int dimensiones) {
        posicion.resize(dimensiones, 0.0); // Usar ceros numéricos 0.0
        m.resize(dimensiones, 0.0);
        v.resize(dimensiones, 0.0);
        gradiente.resize(dimensiones, 0.0); // Corregido 'dimensiones'

        // Posición inicial de la esfera en el tazón
        posicion[0] = 2.0;
        posicion[1] = 2.0;
    }

    void calcularGradiente() {
        double y = posicion[0];
        double x = posicion[1];
        //calculamos el denominador comun del bloque A
        double denominador = std::pow(2, x) + std::pow(2, y);
         
        double bloqueA_Y = 0.0;
        double bloqueA_X = 0.0;

        //Proteccion por si el logaritmo llegara a caerse en (0,0) rompienso el algoritmo llevandolo al
        //infinito rompiendo el algoritmo
        if (denominador > 1e-9){
            bloqueA_Y = (2.0 * x) /denominador;
            bloqueA_X = (2.0 * y) /denominador;

        }
        //Bloque B usando std para la exponencial
        double bloqueB_Y = -y * std::exp(-x * y);
        double bloqueB_X = -x * std::exp(-x * y);
        //ensamblaje final de los bloques para gradiente
        gradiente[0] = bloqueA_Y + bloqueB_Y; //componente y
        gradiente[1] = bloqueA_X + bloqueB_X; //componente x

    }

void actualizarFisica() {
        t++; // Avanzamos el paso de tiempo
        calcularGradiente();

        for (size_t i = 0; i < posicion.size(); ++i) {
            // 1. Actualizar el primer momento móvil (Dirección)
            m[i] = (beta1 * m[i]) + ((1.0 - beta1) * gradiente[i]);

            // 2. Actualizar el segundo momento móvil (Magnitud/Escala)
            v[i] = (beta2 * v[i]) + ((1.0 - beta2) * std::pow(gradiente[i], 2));

            // 3. Cómputo de la corrección de sesgo
            double m_cachucha = m[i] / (1.0 - std::pow(beta1, t));
            double v_cachucha = v[i] / (1.0 - std::pow(beta2, t));

            // 4. Regla de actualización ADAM: divide el paso adaptativamente
            posicion[i] = posicion[i] - (alpha / (std::sqrt(v_cachucha) + epsilon)) * m_cachucha;
        }
    }

   void mostrarEstado() {
    std::cout << "Posicion: [ " << posicion[0] << " " << posicion[1] << " ]"
              << "\tM1 (Inercia): " << m[0]
              << "\tV2 (Escala): " << v[0] << "\n";
}
};

int main() {
    GirosferaEstado miGirosfera(2); // Corregido el tamaño a 2 dimensiones
    
    std::cout << "--- ESTADO INICIAL (Soltando esfera en [x, y]) ---\n";
    miGirosfera.mostrarEstado(); // Corregido typo 'miGirosfera'

    std::cout << "\n--- DESCENDIENDO AL MINIMO GLOBAL ---\n";
    for (int paso = 1; paso <= 50; ++paso) { // 50 pasos son suficientes para ver la convergencia
        miGirosfera.actualizarFisica(); // Corregido typo 'miGirosfera'
        std::cout << "Paso " << paso << " -> "; // Corregido string roto
        miGirosfera.mostrarEstado();
    }
    
    return 0;
}