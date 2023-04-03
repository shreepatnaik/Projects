

#include <SoftwareSerial.h>// import the serial library

#include <dht11.h>
#define DHT11PIN 4
dht11 DHT11;

SoftwareSerial bt(10, 11); // RX, TX
int ledpin=13; // led on D13 will show blink on / off
int BluetoothData; // the data given from Computer
int light;
float temp;
float humidity;
int PowerPin = 3; 
int PowerPin2 = 7;
float moisture; 

void setup() {
  // put your setup code here, to run once:
  bt.begin(9600);
  pinMode(0,INPUT);//light sensor connected to analog 0
  pinMode(5,INPUT);//Moisture sensor connected to analog 0
  analogReference(DEFAULT);
  pinMode(ledpin,OUTPUT);
  pinMode(PowerPin, OUTPUT); 
  pinMode(PowerPin2, OUTPUT);    
  digitalWrite(PowerPin, HIGH); 
  digitalWrite(PowerPin2, HIGH);  

  bt.print("----------WELCOME TO THE SMART GARDEN----------");
  bt.println();

}

void loop() {
  
  
  bt.print("--------------------------------------");
  bt.println();
  int chk = DHT11.read(DHT11PIN);
  
  humidity = DHT11.humidity, 2;
  bt.print("Humidity (%): ");
  bt.println(humidity);

  temp = DHT11.temperature, 2;
  bt.print("Temperature (C): ");
  bt.println(temp);
  
  light = analogRead(0);
  bt.print("Light sensor(-): ");
  bt.println(light); // Reading light sensor
  
  moisture = 100 - analogRead(5)/10;
  bt.print("Moisture sensor(%): ");
  bt.println(moisture);
  
  bt.println();
  
  if(light > 500 or BluetoothData=='0'){
  digitalWrite(ledpin,1);
  bt.println("Lights ON!");
  }
  else if(light <= 499 or BluetoothData=='1'){
  digitalWrite(ledpin,0);
  bt.println("Lights OFF!");
  } 
  if(temp <= 10 or BluetoothData=='2'){
  bt.println("Heat Lamp ON!");
  } 
  else if(temp > 10 or BluetoothData=='3'){
  bt.println("Heat Lamp OFF!");
  } 
  if(humidity <= 20 or BluetoothData=='4' or moisture <= 20 ){
  bt.println("Sprinkler ON!");
  } 
  else if(humidity > 20 or BluetoothData=='5' or moisture > 20){
  bt.println("Sprinkler OFF!");
  } 

  if(BluetoothData=='6'){
  bt.println("Fan ON!");
  } 
  else if(BluetoothData=='7'){
  bt.println("Fan OFF!");
  } 
  
  if (bt.available()){
    
     BluetoothData=bt.read();
  
}
bt.println();
delay(1000);// prepare for next data ...
}
