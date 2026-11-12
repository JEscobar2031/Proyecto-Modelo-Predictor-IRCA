from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse


class RegistrationTests(TestCase):
    def test_registration_page_loads(self):
        response = self.client.get(reverse('predictor:register'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Crear cuenta')

    def test_registration_creates_user_and_redirects_to_login(self):
        response = self.client.post(reverse('predictor:register'), {
            'username': 'profesor',
            'password1': 'AguaSegura2026!',
            'password2': 'AguaSegura2026!',
        })

        self.assertRedirects(response, reverse('predictor:login'))
        self.assertTrue(User.objects.filter(username='profesor').exists())

    def test_registration_rejects_mismatched_passwords(self):
        response = self.client.post(reverse('predictor:register'), {
            'username': 'profesor',
            'password1': 'AguaSegura2026!',
            'password2': 'OtraClave2026!',
        })

        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='profesor').exists())
