from llama_index.core.node_parser import CodeSplitter
from llama_index.core import Document
#just testing CodeSplitter
def test_language(lang):
    try:
        splitter = CodeSplitter(language=lang, chunk_lines=40)
        doc = Document(text="""import kotlin.math.PI

// Define a simple immutable data class
data class Circle(val radius: Double)

// Function to calculate the area of the circle
fun calculateArea(circle: Circle): Double {
    // PI * r^2
    return PI * circle.radius * circle.radius
}

fun main() {
    // 1. Create an instance of the data class
    val myCircle = Circle(radius = 5.0)
    
    // 2. Call the function
    val area = calculateArea(myCircle)

    // 3. Print the result
    println("The circle's radius is: ${myCircle.radius}")
    println("The calculated area is: ${String.format("%.2f", area)}")
}""")
        nodes = splitter.get_nodes_from_documents([doc])
        print(f"✓ {lang} works")
        return True
    except Exception as e:
        print(f"✗ {lang} failed: {e}")
        return False

# Test your key languages
for lang in ["python", "typescript", "java", "cpp", "go", "rust","cpp","kotlin"]:
    test_language(lang)