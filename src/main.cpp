#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "Arduino.h"

// ================= WIFI =================
// const char* ssid = "vivo T3x 5G";
// const char* password = "fabiha123456";
// const char* ssid = "JnU Students";
// const char* password = "stdwifi154";
// const char* ssid = "iPhone";
// const char* password = "11ne2ko13bar14";
// const char* ssid = "Room-601";
// const char* password = "room601601";
const char* ssid = "Galaxy";
const char* password = "11111111";

// ============== CAMERA PINS (AI Thinker) ==============
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define BUZZER_PIN         4   // GPIO 4 for buzzer

httpd_handle_t server = NULL;

// --- Buzzer state ---
bool buzzer_active = false;
unsigned long buzzer_start_time = 0;
const unsigned long BUZZER_DURATION_MS = 3000;

// ================= HTML =================

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-CAM</title>
<style>
html,body{
margin:0;
background:#000;
height:100%;
overflow:hidden;
}
img{
width:100%;
height:100%;
object-fit:contain;
display:block;
}
</style>
</head>
<body>
<img src="/stream">
</body>
</html>
)rawliteral";

esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req,"text/html");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

esp_err_t stream_handler(httpd_req_t *req)
{
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;

    res = httpd_resp_set_type(req,
        "multipart/x-mixed-replace; boundary=frame");

    if(res != ESP_OK)
        return res;

    while(true)
    {
        fb = esp_camera_fb_get();

        if(!fb)
        {
            Serial.println("Camera capture failed");
            continue;
        }

        res = httpd_resp_send_chunk(req,
            "--frame\r\n",
            strlen("--frame\r\n"));

        if(res == ESP_OK)
            res = httpd_resp_send_chunk(req,
            "Content-Type: image/jpeg\r\n\r\n",
            strlen("Content-Type: image/jpeg\r\n\r\n"));

        if(res == ESP_OK)
            res = httpd_resp_send_chunk(req,
            (const char*)fb->buf,
            fb->len);

        if(res == ESP_OK)
            res = httpd_resp_send_chunk(req,"\r\n",2);

        esp_camera_fb_return(fb);

        if(res != ESP_OK)
            break;
    }

    return res;
}

esp_err_t buzzer_handler(httpd_req_t *req)
{
    // Respond immediately, then trigger buzzer non-blocking
    const char* resp = "OK";
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, resp, strlen(resp));

    // Set flag — buzzer will be handled in loop()
    buzzer_active = true;
    buzzer_start_time = millis();
    digitalWrite(BUZZER_PIN, HIGH);
    Serial.println("BUZZER ON");

    return ESP_OK;
}

void startCameraServer()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    config.server_port = 80;
    config.max_uri_handlers = 8;
    config.recv_wait_timeout = 10;
    config.send_wait_timeout = 10;

    httpd_start(&server,&config);

    httpd_uri_t index_uri={
        .uri="/",
        .method=HTTP_GET,
        .handler=index_handler,
        .user_ctx=NULL
    };

    httpd_uri_t stream_uri={
        .uri="/stream",
        .method=HTTP_GET,
        .handler=stream_handler,
        .user_ctx=NULL
    };

    httpd_uri_t buzzer_uri={
        .uri="/buzzer",
        .method=HTTP_GET,
        .handler=buzzer_handler,
        .user_ctx=NULL
    };

    httpd_register_uri_handler(server,&index_uri);
    httpd_register_uri_handler(server,&stream_uri);
    httpd_register_uri_handler(server,&buzzer_uri);
}

void setup()
{
    Serial.begin(115200);
    Serial.println();

    camera_config_t config;

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;

    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;

    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;

    config.pixel_format = PIXFORMAT_JPEG;

    if(psramFound())
    {
        Serial.println("PSRAM Found");

        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 8;
        config.fb_count = 2;
        config.grab_mode = CAMERA_GRAB_LATEST;
    }
    else
    {
        Serial.println("No PSRAM");

        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 10;
        config.fb_count = 1;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    }

    esp_err_t err = esp_camera_init(&config);

    if(err != ESP_OK)
    {
        Serial.printf("Camera Init Failed : 0x%x\n",err);
        return;
    }

    sensor_t * s = esp_camera_sensor_get();

    s->set_brightness(s,0);
    s->set_contrast(s,0);
    s->set_saturation(s,0);
    s->set_sharpness(s,1);
    s->set_framesize(s,FRAMESIZE_QVGA);

    // Initialize buzzer pin
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid,password);

    Serial.print("Connecting");

    while(WiFi.status()!=WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    WiFi.setSleep(false);
    WiFi.setTxPower(WIFI_POWER_19_5dBm);

    Serial.println();
    Serial.println("WiFi Connected");
    Serial.print("IP : ");
    Serial.println(WiFi.localIP());

    startCameraServer();

    Serial.println();
    Serial.print("Open : http://");
    Serial.println(WiFi.localIP());
}

void loop()
{
    // Non-blocking buzzer control
    if (buzzer_active && (millis() - buzzer_start_time >= BUZZER_DURATION_MS))
    {
        digitalWrite(BUZZER_PIN, LOW);
        buzzer_active = false;
        Serial.println("BUZZER OFF");
    }
}