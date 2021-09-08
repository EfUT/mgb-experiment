from django.shortcuts import render
from .forms import ImageForm
import uuid


def upload_file(request):
    if request.method == 'POST':
        print(request.POST)
        form = ImageForm(request.POST, request.FILES)
        form.initial['title'] = str(uuid.uuid4())
        if form.is_valid():
            form.save()
            img_obj = form.instance
            return render(request, 'index.html', {
                'form': form,
                'img_obj': img_obj
            })
        else:
            print(form.errors)
    else:
        form = ImageForm()
    return render(request, 'index.html', context={'form': form, "title": 'aaa'})