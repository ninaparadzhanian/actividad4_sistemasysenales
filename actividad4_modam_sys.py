import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class SistemaModulacionAM:
    """
    Sistema completo de Modulación de Amplitud (AM)
    Incluye generación de señal, modulación, análisis y efectos del ruido
    """
    
    def __init__(self, fs=10000, duracion=1.0):
        """
        Inicializa el sistema de modulación
        
        Parámetros:
        - fs: Frecuencia de muestreo (Hz)
        - duracion: Duración de la señal (segundos)
        """
        self.fs = fs
        self.duracion = duracion
        self.t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
        self.N = len(self.t)
        
    def crear_senal_mensaje(self, tipo='compuesta', **kwargs):
        """
        Crea la señal de información (baja frecuencia)
        
        Tipos disponibles:
        - 'simple': Senoidal pura
        - 'compuesta': Múltiples frecuencias
        - 'cuadrada': Onda cuadrada
        - 'triangular': Onda triangular
        - 'audio': Simulación de señal de audio
        """
        if tipo == 'simple':
            freq = kwargs.get('freq', 10)
            amplitud = kwargs.get('amplitud', 1.0)
            mensaje = amplitud * np.sin(2 * np.pi * freq * self.t)
            info = f"Senoidal {freq} Hz"
            
        elif tipo == 'compuesta':
            freqs = kwargs.get('freqs', [5, 15, 25])
            amps = kwargs.get('amps', [1.0, 0.5, 0.3])
            mensaje = sum(a * np.sin(2 * np.pi * f * self.t) for f, a in zip(freqs, amps))
            info = f"Compuesta {freqs} Hz"
            
        elif tipo == 'cuadrada':
            freq = kwargs.get('freq', 10)
            amplitud = kwargs.get('amplitud', 1.0)
            mensaje = amplitud * signal.square(2 * np.pi * freq * self.t)
            info = f"Cuadrada {freq} Hz"
            
        elif tipo == 'triangular':
            freq = kwargs.get('freq', 10)
            amplitud = kwargs.get('amplitud', 1.0)
            mensaje = amplitud * signal.sawtooth(2 * np.pi * freq * self.t, width=0.5)
            info = f"Triangular {freq} Hz"
            
        elif tipo == 'audio':
            # Simula una señal de audio con varias componentes
            freqs = [10, 20, 30, 50]
            amps = [1.0, 0.6, 0.4, 0.2]
            mensaje = sum(a * np.sin(2 * np.pi * f * self.t) for f, a in zip(freqs, amps))
            # Agregar envolvente para simular dinámica de audio
            envolvente = np.exp(-2 * self.t) * np.sin(2 * np.pi * 0.5 * self.t)**2
            mensaje = mensaje * envolvente
            info = "Simulación de Audio"
            
        return mensaje, info
    
    def modular_am(self, mensaje, fc=1000, indice_modulacion=0.8):
        """
        Realiza la modulación AM
        
        Parámetros:
        - mensaje: Señal de información
        - fc: Frecuencia de la portadora (Hz)
        - indice_modulacion: Índice de modulación (0-1)
        
        Retorna:
        - portadora: Señal portadora
        - senal_modulada: Señal AM modulada
        """
        # Normalizar mensaje para evitar sobremodulación
        mensaje_norm = mensaje / np.max(np.abs(mensaje))
        
        # Crear portadora
        portadora = np.cos(2 * np.pi * fc * self.t)
        
        # Modulación AM: s(t) = [1 + m * x(t)] * cos(2π fc t)
        senal_modulada = (1 + indice_modulacion * mensaje_norm) * portadora
        
        return portadora, senal_modulada, mensaje_norm
    
    def agregar_ruido(self, senal, snr_db=20):
        """
        Agrega ruido blanco gaussiano a la señal
        
        Parámetros:
        - senal: Señal original
        - snr_db: Relación Señal-Ruido en dB
        
        Retorna:
        - senal_ruidosa: Señal con ruido
        - ruido: Componente de ruido agregado
        """
        # Calcular potencia de la señal
        potencia_senal = np.mean(senal ** 2)
        
        # Calcular potencia del ruido según SNR
        snr_lineal = 10 ** (snr_db / 10)
        potencia_ruido = potencia_senal / snr_lineal
        
        # Generar ruido blanco gaussiano
        ruido = np.sqrt(potencia_ruido) * np.random.randn(len(senal))
        
        # Agregar ruido a la señal
        senal_ruidosa = senal + ruido
        
        return senal_ruidosa, ruido
    
    def analizar_espectro(self, senal):
        """
        Calcula el espectro de frecuencias de una señal
        """
        # Calcular FFT
        espectro = fft(senal)
        freqs = fftfreq(self.N, 1/self.fs)
        
        # Magnitud del espectro (valores positivos)
        magnitud = np.abs(espectro) / self.N
        
        return freqs, magnitud
    
    def calcular_metricas_calidad(self, senal_original, senal_degradada):
        """
        Calcula métricas de calidad de la señal
        """
        # Error cuadrático medio
        mse = np.mean((senal_original - senal_degradada) ** 2)
        
        # Relación Señal-Ruido
        potencia_original = np.mean(senal_original ** 2)
        potencia_ruido = np.mean((senal_original - senal_degradada) ** 2)
        snr = 10 * np.log10(potencia_original / potencia_ruido)
        
        # Correlación
        correlacion = np.corrcoef(senal_original, senal_degradada)[0, 1]
        
        return {
            'MSE': mse,
            'SNR_dB': snr,
            'Correlación': correlacion
        }
    
    def graficar_senal_tiempo(self, t, senal, titulo, ax=None, color='b'):
        """
        Grafica una señal en el dominio del tiempo
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(t, senal, color, linewidth=1)
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Amplitud')
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([t[0], min(t[-1], 0.05)])  # Mostrar primeros 50ms
        
        return ax
    
    def graficar_espectro(self, freqs, magnitud, titulo, ax=None, color='r', 
                          xlim=None):
        """
        Grafica el espectro de frecuencias
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        
        # Solo frecuencias positivas
        mask = freqs >= 0
        freqs_pos = freqs[mask]
        magnitud_pos = magnitud[mask]
        
        ax.plot(freqs_pos, magnitud_pos, color, linewidth=1.5)
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Magnitud')
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
        
        if xlim:
            ax.set_xlim(xlim)
        
        return ax
    
    def visualizar_modulacion_completa(self, mensaje, portadora, modulada, 
                                      info_mensaje, fc, indice_mod):
        """
        Visualización completa del proceso de modulación
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Señal mensaje (tiempo)
        ax1 = plt.subplot(3, 3, 1)
        self.graficar_senal_tiempo(self.t, mensaje, 
                                  f'Señal Mensaje - {info_mensaje}', ax1, 'b')
        
        # 2. Señal mensaje (frecuencia)
        ax2 = plt.subplot(3, 3, 2)
        freqs_msg, mag_msg = self.analizar_espectro(mensaje)
        self.graficar_espectro(freqs_msg, mag_msg, 
                              'Espectro del Mensaje', ax2, 'b', [0, 100])
        
        # 3. Información de parámetros
        ax3 = plt.subplot(3, 3, 3)
        ax3.axis('off')
        info_text = f"""
        PARÁMETROS DEL SISTEMA
        ━━━━━━━━━━━━━━━━━━━━━━━
        • Frecuencia de muestreo: {self.fs} Hz
        • Duración: {self.duracion} s
        • Frecuencia portadora: {fc} Hz
        • Índice de modulación: {indice_mod}
        • Tipo de mensaje: {info_mensaje}
        """
        ax3.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 4. Portadora (tiempo)
        ax4 = plt.subplot(3, 3, 4)
        self.graficar_senal_tiempo(self.t, portadora, 
                                  f'Portadora - {fc} Hz', ax4, 'g')
        
        # 5. Portadora (frecuencia)
        ax5 = plt.subplot(3, 3, 5)
        freqs_port, mag_port = self.analizar_espectro(portadora)
        self.graficar_espectro(freqs_port, mag_port, 
                              'Espectro de Portadora', ax5, 'g', [0, 2000])
        
        # 6. Señal Modulada (tiempo)
        ax6 = plt.subplot(3, 3, 6)
        self.graficar_senal_tiempo(self.t, modulada, 
                                  'Señal Modulada AM', ax6, 'r')
        
        # 7. Señal Modulada (frecuencia - vista general)
        ax7 = plt.subplot(3, 3, 7)
        freqs_mod, mag_mod = self.analizar_espectro(modulada)
        self.graficar_espectro(freqs_mod, mag_mod, 
                              'Espectro de Señal Modulada', ax7, 'r', [0, 2000])
        
        # 8. Zoom del espectro modulado
        ax8 = plt.subplot(3, 3, 8)
        self.graficar_espectro(freqs_mod, mag_mod, 
                              'Zoom - Bandas Laterales', ax8, 'r', 
                              [fc-100, fc+100])
        
        # 9. Comparación temporal
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(self.t[:500], mensaje[:500], 'b-', label='Mensaje', alpha=0.7)
        ax9.plot(self.t[:500], modulada[:500], 'r-', label='Modulada', alpha=0.7)
        ax9.set_xlabel('Tiempo (s)')
        ax9.set_ylabel('Amplitud')
        ax9.set_title('Comparación Mensaje vs Modulada')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle(f'SISTEMA DE MODULACIÓN AM - Análisis Completo', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def analizar_efectos_ruido(self, modulada_original, fc, snr_levels=[30, 20, 10, 5, 0]):
        """
        Analiza el efecto del ruido en la señal modulada a diferentes SNR
        """
        fig = plt.figure(figsize=(16, 12))
        
        metricas_todos = []
        
        for i, snr in enumerate(snr_levels):
            # Agregar ruido
            modulada_ruidosa, ruido = self.agregar_ruido(modulada_original, snr)
            
            # Calcular métricas
            metricas = self.calcular_metricas_calidad(modulada_original, modulada_ruidosa)
            metricas['SNR_objetivo'] = snr
            metricas_todos.append(metricas)
            
            # Graficar señal ruidosa
            ax1 = plt.subplot(len(snr_levels), 3, 3*i + 1)
            self.graficar_senal_tiempo(self.t, modulada_ruidosa, 
                                      f'SNR = {snr} dB', ax1, 'purple')
            
            # Graficar espectro
            ax2 = plt.subplot(len(snr_levels), 3, 3*i + 2)
            freqs, mag = self.analizar_espectro(modulada_ruidosa)
            self.graficar_espectro(freqs, mag, 
                                  f'Espectro SNR={snr} dB', ax2, 'purple', 
                                  [fc-200, fc+200])
            
            # Mostrar métricas
            ax3 = plt.subplot(len(snr_levels), 3, 3*i + 3)
            ax3.axis('off')
            metrics_text = f"""
            MÉTRICAS SNR={snr}dB
            ━━━━━━━━━━━━━━━━━━
            MSE: {metricas['MSE']:.4f}
            SNR real: {metricas['SNR_dB']:.2f} dB
            Correlación: {metricas['Correlación']:.3f}
            """
            ax3.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='center')
        
        plt.suptitle('EFECTOS DEL RUIDO EN LA SEÑAL MODULADA', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, metricas_todos
    
    def analizar_escenarios(self, modulada_original, fc):
        """
        Analiza diferentes escenarios de distorsión y atenuación
        """
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        
        escenarios = [
            {
                'nombre': 'Señal Original',
                'senal': modulada_original,
                'color': 'blue'
            },
            {
                'nombre': 'Atenuación 50%',
                'senal': modulada_original * 0.5,
                'color': 'green'
            },
            {
                'nombre': 'Recorte (Clipping)',
                'senal': np.clip(modulada_original, -0.5, 0.5),
                'color': 'red'
            },
            {
                'nombre': 'Ruido SNR=15dB',
                'senal': self.agregar_ruido(modulada_original, 15)[0],
                'color': 'purple'
            },
            {
                'nombre': 'Interferencia 50Hz',
                'senal': modulada_original + 0.3*np.sin(2*np.pi*50*self.t),
                'color': 'orange'
            },
            {
                'nombre': 'Distorsión no lineal',
                'senal': np.tanh(modulada_original * 2),
                'color': 'brown'
            }
        ]
        
        for i, escenario in enumerate(escenarios):
            row = i // 3
            col = i % 3
            
            # Graficar en tiempo
            ax = axes[row, col]
            ax.plot(self.t[:500], escenario['senal'][:500], 
                   color=escenario['color'], linewidth=1)
            ax.set_title(f"{escenario['nombre']}")
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.grid(True, alpha=0.3)
            
            # Calcular y mostrar métricas si no es original
            if i > 0:
                metricas = self.calcular_metricas_calidad(modulada_original, 
                                                         escenario['senal'])
                text = f"SNR: {metricas['SNR_dB']:.1f}dB\nCorr: {metricas['Correlación']:.3f}"
                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('ANÁLISIS DE DISTORSIÓN Y ATENUACIÓN - DIFERENTES ESCENARIOS',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

# Función principal de demostración
def demostracion_sistema_modulacion():
    """
    Demostración completa del sistema de modulación AM
    """
    print("=" * 70)
    print("SISTEMA DE MODULACIÓN DE AMPLITUD (AM) - DEMOSTRACIÓN COMPLETA")
    print("=" * 70)
    
    # Crear sistema
    sistema = SistemaModulacionAM(fs=20000, duracion=0.5)
    
    # 1. Crear señal mensaje
    print("\n1. Generando señal de información...")
    mensaje, info = sistema.crear_senal_mensaje(tipo='compuesta', 
                                               freqs=[10, 25, 40],
                                               amps=[1.0, 0.6, 0.3])
    print(f"   Tipo de mensaje: {info}")
    
    # 2. Realizar modulación AM
    print("\n2. Realizando modulación AM...")
    fc = 2000  # Frecuencia portadora
    indice_mod = 0.8
    portadora, modulada, mensaje_norm = sistema.modular_am(mensaje, fc, indice_mod)
    print(f"   Frecuencia portadora: {fc} Hz")
    print(f"   Índice de modulación: {indice_mod}")
    
    # 3. Visualizar proceso completo
    print("\n3. Visualizando proceso de modulación...")
    fig_modulacion = sistema.visualizar_modulacion_completa(
        mensaje, portadora, modulada, info, fc, indice_mod)
    plt.show()
    
    # 4. Analizar efectos del ruido
    print("\n4. Analizando efectos del ruido...")
    fig_ruido, metricas = sistema.analizar_efectos_ruido(modulada, fc)
    plt.show()
    
    print("\n   Métricas de degradación por ruido:")
    for m in metricas:
        print(f"   SNR objetivo: {m['SNR_objetivo']:2d} dB → "
              f"SNR real: {m['SNR_dB']:6.2f} dB, "
              f"Correlación: {m['Correlación']:.3f}")
    
    # 5. Analizar diferentes escenarios
    print("\n5. Analizando escenarios de distorsión...")
    fig_escenarios = sistema.analizar_escenarios(modulada, fc)
    plt.show()
    
    # 6. Análisis adicional: Variación del índice de modulación
    print("\n6. Analizando efecto del índice de modulación...")
    indices = [0.3, 0.6, 0.9, 1.2]  # 1.2 causa sobremodulación
    
    fig_indices, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, im in enumerate(indices):
        row = i // 2
        col = i % 2
        
        _, modulada_im, _ = sistema.modular_am(mensaje, fc, im)
        
        # Graficar
        axes[row, col].plot(sistema.t[:500], modulada_im[:500], 'b-', linewidth=1)
        axes[row, col].plot(sistema.t[:500], mensaje_norm[:500], 'r--', 
                           linewidth=1, alpha=0.5, label='Mensaje')
        
        estado = "SOBREMODULACIÓN" if im > 1 else "Normal"
        color_title = 'red' if im > 1 else 'black'
        axes[row, col].set_title(f'Índice de Modulación = {im} ({estado})', 
                                color=color_title, fontweight='bold')
        axes[row, col].set_xlabel('Tiempo (s)')
        axes[row, col].set_ylabel('Amplitud')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend()
    
    plt.suptitle('EFECTO DEL ÍNDICE DE MODULACIÓN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return sistema

# Ejemplo de uso específico con diferentes tipos de señales
def ejemplos_adicionales():
    """
    Ejemplos adicionales con diferentes tipos de señales
    """
    sistema = SistemaModulacionAM(fs=10000, duracion=0.3)
    fc = 1500
    
    tipos_senales = ['simple', 'cuadrada', 'triangular', 'audio']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    for i, tipo in enumerate(tipos_senales):
        # Crear mensaje
        if tipo == 'simple':
            mensaje, info = sistema.crear_senal_mensaje(tipo='simple', freq=20)
        else:
            mensaje, info = sistema.crear_senal_mensaje(tipo=tipo)
        
        # Modular
        _, modulada, _ = sistema.modular_am(mensaje, fc, 0.7)
        
        # Graficar mensaje
        ax1 = axes[0, i]
        ax1.plot(sistema.t[:300], mensaje[:300], 'b-', linewidth=1.5)
        ax1.set_title(f'Mensaje: {info}')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud')
        ax1.grid(True, alpha=0.3)
        
        # Graficar modulada
        ax2 = axes[1, i]
        ax2.plot(sistema.t[:300], modulada[:300], 'r-', linewidth=1)
        ax2.set_title(f'Modulada AM - {info}')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud')
        ax2.grid(True, alpha=0.3)
    
    # Ocultar ejes vacíos si los hay
    for j in range(len(tipos_senales), 4):
        axes[0, j].axis('off')
        axes[1, j].axis('off')
    
    plt.suptitle('MODULACIÓN AM CON DIFERENTES TIPOS DE SEÑALES', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Ejecutar demostración
if __name__ == "__main__":
    # Demostración principal
    sistema = demostracion_sistema_modulacion()
    
    # Ejemplos adicionales
    print("\n" + "=" * 70)
    print("EJEMPLOS ADICIONALES CON DIFERENTES SEÑALES")
    print("=" * 70)
    ejemplos_adicionales()