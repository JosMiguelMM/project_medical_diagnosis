from django.shortcuts import render
from django.http import HttpResponse
from .clustering import KMeansCluster  # Importa desde clustering.py
import pandas as pd
from django.core.files.storage import FileSystemStorage

def clustering_view(request):
    if request.method == 'POST' and request.FILES.get('datafile'):
        myfile = request.FILES['datafile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(fs.path(filename))
            elif filename.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(fs.path(filename))
            else:
                return HttpResponse("Formato de archivo no soportado.", status=400)

            data_cleaned = data.dropna()
            numerical_cols = data_cleaned.select_dtypes(include=['number']).columns
            data_for_clustering = data_cleaned[numerical_cols]

            num_clusters = int(request.POST.get('num_clusters', 3))
            kmeans = KMeansCluster(n_clusters=num_clusters)
            kmeans.fit(data_for_clustering)
            cluster_centers = kmeans.get_cluster_centers()
            labels = kmeans.get_labels()

            data_cleaned['cluster'] = labels
            data_with_clusters = data_cleaned.to_dict('records')

            context = {
                'uploaded_file_url': uploaded_file_url,
                'cluster_centers': cluster_centers.tolist(),
                'data_with_clusters': data_with_clusters,
                'labels': labels.tolist(),
                'success': True,
            }
            return render(request, 'clustering_results.html', context)  # Template en la raíz

        except Exception as e:
            return render(request, 'clustering_results.html', {'error': str(e)})

    return render(request, 'clustering_results.html')  # Template en la raíz
