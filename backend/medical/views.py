# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
# def process_data(request):
#     if request.method == "POST":
#         query = request.POST.get('query')
#         if not query:
#             return JsonResponse({"error": "Query parameter is missing."}, status=400)

#         file = request.FILES.get('file')
#         if file:
#             print(f"Received file: {file.name}, Size: {file.size} bytes")

#         return JsonResponse({"message": f"Received query: {query}"})
#     return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def process_data(request):
    if request.method == "POST":
        query = request.POST.get('query')
        if not query:
            return JsonResponse({"error": "Query parameter is missing."}, status=400)

        file = request.FILES.get('file')
        if file:
            print(f"Received file: {file.name}, Size: {file.size} bytes")

        return JsonResponse({"message": f"Received query: {query}"})
    return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)


