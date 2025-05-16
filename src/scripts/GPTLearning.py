from openai import OpenAI
import re
import os
from tqdm import tqdm
from pydantic import BaseModel, Field
import pandas as pd

# Modelo para validar la salida
class CognitiveOutput(BaseModel):
    cognitiveDet: bool
    cognitiveRanking: int = Field(..., ge=0, le=4)

class GPTLearning:
    def __init__(self):
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
            
        self.client = OpenAI(api_key=api_key)
        self.results = []
        self.prompt_text = """

            Analiza la entrevista de una persona mayor en la que se le hacen preguntas sobre lo que ven en una foto, lo que ven en un video y un recuerdo de su lugar favorito. A continuación, categoriza si la persona tiene deterioro cognitivo o no y asigna un grado de deterioro entre 0 y 4, donde 0 indica deterioro mínimo o no aparente y 4 indica un deterioro significativo. Razona cada clasificación antes de llegar a una conclusión.

            ## Detalles adicionales

            - **Evaluación de la descripción de la foto y video**: Observa la claridad, detalle, y coherencia en la explicación de lo que se ve, buscando signos de confusión o falta de reconocimiento.
            - **Recuerdo del lugar favorito**: Analiza la capacidad del individuo para recordar detalles específicos, emocionalidad, y coherencia.
            - **Foto mostrada**: Una mujer en su pusto de trabajo agobiada con muchas carpetas y archivadores en la mesa.
            - **Video mostrado**: Un pez es atrapado para ser cocinado. Este intenta escaparse y se dirige a un cuadro del mar pensado que es mar realmente. Al chocarse con el cuadro, acabo siendo cocinado.

            # Output Format

            Respuesta estructurada en un párrafo que incluya el razonamiento seguido y la categorización final. Especifica el grado de deterioro de manera clara.
            **Importante**: Finaliza la respuesta con la frase: "Su grado de deterioro se clasifica como: X", donde X es un número del 0 al 4.

            # Ejemplo Grupo 0

            ### Entrada:
            Pues que tiene un problema. puede ser de cálculo o de exceso de trabajo porque tiene muchos papeles.Un pez. el pobrecito que está tan tranquilo en el agua. llega un cocinero. le saca por la cola. le quiere echar a la sartén y él se defiende. se infla como es un pez globo. quiere salir. le quieren pillar por la ventana. con unas ventanas correderas y al final se desinfla y se cae el sol en la sartén o en la cacerola o lo que sea.Siempre habría deseado hacer un viaje a Japón. Pensé que cuando me divorcie. cuando me jubile. me voy a Japón. pero por cuestiones familiares no pude. Siempre me he quedado con ganas de conocer Japón.

            ### Salida:
            La persona describió la foto mostrada con claridad, incluyendo ideas como el problema de cáculo. El video lo describle con claridad y bastantes detalles, tanto de la escena como del pez. En el recuerdo del lugar favorito, la persona muestra una conexión emocional y un deseo de viajar a Japón, aunque no pudo hacerlo. Basado en estas observaciones, no parece que la persona tenga un deterioro cognitivo significativo. Su grado de deterioro se clasifica como 0, indicando un deterioro mínimo o no aparente.


            # Ejemplo Grupo 1

            ### Entrada:
            Pues es una mujer que está agobiada de trabajo. se ve clarísimo. tiene ahí un montón de trabajo pendiente y no le debe dar tiempo a hacer todo lo que tiene ahí pendiente.No está muy contenta porque además tiene pinta de ser una mujer. un ejecutivo. de ser una ejecutiva que va con traje y todo.Es un pez que está tan tranquilo ahí y le coge un cocinero por la cola y tiene preparado el fuego con el aceite y lo va a echar.Y llega un momento en que llega. hace un esfuerzo y pasa por la puerta y sale volando y va a una ola. que esa ola además es muy conocida. es un pintor.sí. sería mi lugar favorito porque me mola en todas las condiciones está tranquila tiene el clima adecuado que me gusta no es ni calor ni frío es muy bonito muy pacífico y tiene realmente para vivir el círculo de los alimentantes de los peces que pesques y de la fruta. de los árboles con lo cual no necesitas nada la ropa. que te haga cualquier cosa te haces con una hoja o algo de eso un trenzado de eso con hojas y ya está.

            ### Salida:
            La intervención muestra un discurso relativamente comprensible, con un lenguaje sencillo y un contenido que, aunque algo desorganizado en su expresión, refleja intención comunicativa clara y asociaciones lógicas dentro de cada respuesta. Se observan ideas completas aunque con cierta pobreza gramatical y ocasionales repeticiones o frases entrecortadas, lo que podría sugerir una ligera dificultad en la fluidez verbal. No hay alteraciones graves en la comprensión ni en el contenido de pensamiento, pero sí se perciben signos sutiles de lentitud cognitiva o esfuerzo en la elaboración de las ideas. Estos indicadores son compatibles con un grado muy leve de deterioro cognitivo (Grupo 1), caracterizado por una preservación general de las funciones, aunque con posibles afectaciones iniciales en la planificación del discurso y la atención sostenida.


            # Ejemplo Grupo 2
            ### Entrada:
            Pues una persona que está con libros y con cuadernos y con cosas así. que hay carteras y que tiene de todo.He visto un cocinero que estaba esperando un pez. que no sé qué pez era. lo quería guisar y se lo escapaba. me imagino que sea así. pero nada más.Tenía que terminar en la cazuela.Me gustan mucho las flores. pues en un jardín de flores que me gustan.Un jardín con muchas plantas y de todos los colores.Pues tiene una entrada. o sea. en el portal. ¿no? O ya de la puerta de mi casa para adentro.Tiene un armario desde una puerta hasta la otra puerta.

            ### Salida:
            La intervención presenta un discurso fragmentado, con frases breves, escasa cohesión y dificultades para desarrollar ideas de forma continua, lo que sugiere un grado leve de deterioro cognitivo compatible con el Grupo 2. Aunque las respuestas contienen palabras con sentido y algunos elementos descriptivos, hay una clara pobreza en la estructuración del contenido, repeticiones innecesarias y dificultades para mantener una narrativa coherente. El pensamiento parece concretarse en imágenes o escenas sueltas, sin elaboración ni conexión entre ellas, y con una evidente limitación en la capacidad de organización verbal y planificación del discurso. Estos signos indican una afectación más notable que en el grupo 1, pero sin llegar a una desestructuración severa, siendo indicativos de un deterioro leve pero más evidente en funciones ejecutivas y memoria de trabajo verbal.


            # Ejemplo Grupo 3
            ### Entrada:
            Pues una oficina con muchas carpetas y una señorita que no sabe por dónde empezar a colocarlas.Se siente agobiada por todo el trabajo que se le ha acumulado.Pues he visto un pez. un cocinero. con un cuchillo. una tabla de cortar. un bol.En el vídeo le iba a coger y se escapó. Se ha hinchado y quería salir.El mío. la casa.Yo he trabajado fuera y he estado muchas horas fuera de casa y entonces ahora me apetece mucho estar en casa.Y estar con mis hijos y mis nietas.

            ### Salida:
            La intervención revela un discurso más limitado en elaboración y profundidad, con frases breves, escasa elaboración temática y una tendencia a describir elementos aislados sin conectar claramente las ideas, lo que es característico de un deterioro cognitivo moderado correspondiente al Grupo 3. Aunque el contenido conserva cierta lógica interna, se observa reducción en la riqueza del lenguaje, dificultad para construir narrativas más complejas y una dependencia de elementos visuales concretos o emocionales, sin capacidad para desarrollarlos de forma más abstracta o extensa. Las respuestas tienden a ser simples, con predominio de afirmaciones directas y sin articulación entre escenas o temas, lo que sugiere déficits en la memoria operativa, la planificación discursiva y la flexibilidad cognitiva, signos compatibles con una alteración moderada de las funciones ejecutivas y del lenguaje espontáneo.


            # Ejemplo Grupo 4
            ### Entrada:
            No. la veo un poco triste.Es de oficina porque tiene muchos libros y está pensando en lo que hace. lo que tiene que hacer o lo que debe de hacer.Pues un montón de carpetas ahí puestas y una señora así.Pues va el pez andando de un lado para el otro. el cocinero lo ve. no lo puede coger y...Pues puede huir. correr para que no lo coja. escapase.Pues andando como peces cada uno. se escapa para que no lo coja. no lo coja el cocinero.Yo entro en el salón. es la entrada al hall. luego así entra a estar el salón. aquí pasa a estar el cuarto de estar. que antes era el dormitorio y ahora está bombado. porque ya no tenemos cuarto de estar.Y luego del hall pasa. hay un baño. luego sigue para adentro. está así la habitación de él. más adentro está la cocina. más adentro está mi dormitorio y luego otro baño.Porque mi casa. llevo ahí muchos años. no han nacido mis hijos ahí. pero llevamos ahí muchos años.Hombre. a mí me gusta mucho vivir en Madrid.Y eso. cuando vamos. a lo mejor me voy al año. que a todos los otros años me gusta pasar.Y hace poco me pasé mi casa por donde vivía y todo. eso me hizo mucha ilusión.

            ### Salida:
            La intervención muestra un discurso claramente desorganizado, con dificultades marcadas en la estructuración de ideas, uso de un lenguaje empobrecido y tendencia a la repetición y perseveración, lo que sugiere un deterioro cognitivo moderado a severo, correspondiente al Grupo 4. Las respuestas presentan frases incompletas, conexiones ilógicas o ambiguas entre conceptos, y una notable pérdida de cohesión discursiva, con un pensamiento que se dispersa fácilmente y se enreda en descripciones vagas o redundantes. Aunque se conservan ciertos contenidos emocionales y referencias personales, la planificación narrativa está deteriorada, hay confusión en la organización espacial y temporal, y se evidencian alteraciones significativas en funciones ejecutivas, lenguaje y memoria de trabajo, lo que indica una afectación más profunda de las capacidades cognitivas.


            # Notas:

            - Ten en cuenta la capacidad del sujeto para mantenerse en el tema, la consistencia de detalles y la claridad del discurso.
            - La emotividad y la conexión personal con los recuerdos pueden influir en la evaluación del deterioro.
            - Contextualiza siempre dentro de lo que se considera común para la edad del entrevistado.
            - Ten en cuenta que el texto de entrada son fragmentos de entrevistas diferentes del mismo paciente y no un texto completo, por eso el cambio de temas y falta de coherencia.

            """        

    
    def analizar_entrevistas(self, grouped):
        for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
            text = row['Sentence']
            codigo = row['CodigoSujeto']
            grupo = row['Grupo']

            input_text = self.prompt_text + "\n\n" + text

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": input_text}],
                    temperature=1,
                    max_tokens=2048,
                    top_p=1
                )
                output_text = response.choices[0].message.content

                print(output_text)

                # Buscar el grado de deterioro
                ranking_match = re.search(r'(grado de deterioro|grupo|clasificación).*?(\d)', output_text, re.IGNORECASE)
                ranking = int(ranking_match.group(2)) if ranking_match else -1

                det = ranking > 0 if ranking != -1 else None



                if ranking != -1:
                    cognitive_output = CognitiveOutput(cognitiveDet=det, cognitiveRanking=ranking)

                    self.results.append({
                        "Text": text,
                        "CodigoSujeto": codigo,
                        "Grupo": grupo,
                        "cognitiveDet": cognitive_output.cognitiveDet,
                        "predicted_label": cognitive_output.cognitiveRanking
                    })

                else:
                    print(f"No se encontró un número de deterioro en la respuesta del modelo para {codigo}.\nTexto:\n{output_text}\n")

                    self.results.append({
                        "Text": text,
                        "CodigoSujeto": codigo,
                        "Grupo": grupo,
                        "cognitiveDet": None,
                        "predicted_label": None
                    })


            except Exception as e:
                print(f"Error procesando {codigo}: {e}")
                self.results.append({
                    "Text": text,
                    "CodigoSujeto": codigo,
                    "Grupo": grupo,
                    "cognitiveDet": None,
                    "predicted_label": None
                })
    

    def save_results(self, output_path):
        final_df = pd.DataFrame(self.results)
        final_df.to_csv(output_path, sep="\t", index=False)

    
df = pd.read_csv("./data/only_sentences.tsv", sep="\t")
GPTLearning = GPTLearning()
GPTLearning.analizar_entrevistas(df)
GPTLearning.save_results("./predictions/output_cognitive_gpt.tsv")