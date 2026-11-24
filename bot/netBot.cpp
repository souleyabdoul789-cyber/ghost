#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <chrono>
using namespace std;

void telecharger(string lien) {
    string cmd = "yt-dlp \"" + lien + "\"";
    cout << "[BOT] Je telecharge : " << lien << endl;
    system(cmd.c_str());
    cout << "[BOT] Fini !" << endl;
}

void surveiller(string lien, int minutes) {
    cout << "[BOT] Je surveille : " << lien << " toutes les " << minutes << " min" << endl;
    string cmd1 = "curl -s \"" + lien + "\" > avant.txt";
    system(cmd1.c_str());

    while (true) {
        this_thread::sleep_for(chrono::minutes(minutes));
        system(("curl -s \"" + lien + "\" > apres.txt").c_str());

        int resultat = system("diff -q avant.txt apres.txt > /dev/null");
        if (resultat != 0) {
            cout << "\n[ALERTE] CA A CHANGE ! " << lien << endl;
            cout << "[BOT] Je telecharge la nouvelle version..." << endl;
            system("cp apres.txt avant.txt");
        } else {
            cout << "[BOT] Rien de nouveau..." << endl;
        }
    }
}

int main() {
    int choix;
    string lien;
    int minutes;

    while (true) {
        cout << "\n=== MON BOT ===" << endl;
        cout << "1. Telecharger" << endl;
        cout << "2. Surveiller un site" << endl;
        cout << "3. Quitter" << endl;
        cout << "Choix : ";
        cin >> choix;
        cin.ignore();

        if (choix == 1) {
            cout << "Colle le lien : ";
            getline(cin, lien);
            telecharger(lien);
        }
        else if (choix == 2) {
            cout << "Colle le lien a surveiller : ";
            getline(cin, lien);
            cout << "Verifier toutes les combien de minutes ? ";
            cin >> minutes;
            surveiller(lien, minutes);
        }
        else break;
    }
    return 0;
}
