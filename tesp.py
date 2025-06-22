from tools import TOOLS

# Simulasi pemanggilan tool get_most_loved tanpa argumen
try:
    result = TOOLS["get_most_loved"].invoke({})
    print("=== HASIL TOOL get_most_loved ===")
    print(result)
except Exception as e:
    print("=== GAGAL MEMANGGIL TOOL ===")
    print(e)