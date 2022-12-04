from rest_framework.generics import RetrieveUpdateDestroyAPIView, ListCreateAPIView
from rest_framework.permissions import IsAuthenticated
from django_filters import rest_framework as filters
from .models import FacePicture
from .permissions import IsOwnerOrReadOnly
from .serializers import FacePictureSerializer
from .pagination import CustomPagination
from .filters import FacePictureFilter



class ListCreateMovieAPIView(ListCreateAPIView):
    serializer_class = FacePictureSerializer
    queryset = FacePicture.objects.all()
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination
    filter_backends = (filters.DjangoFilterBackend,)
    filterset_class = FacePictureFilter

    def perform_create(self, serializer):
        # Assign the user who created the movie
        serializer.save(creator=self.request.user)


class RetrieveUpdateDestroyMovieAPIView(RetrieveUpdateDestroyAPIView):
    serializer_class = FacePictureSerializer
    queryset = FacePicture.objects.all()
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]




